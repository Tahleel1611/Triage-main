"""
Stress Test Suite for Triage HMS API
=====================================
Tests robustness against real-world Emergency Department scenarios:
1. Partial Vitals (missing data → median imputation)
2. High-Load Simulation (concurrent requests → SSE/BackgroundTask bottlenecks)
3. Invalid Inputs (extreme outliers → preprocessor resilience)

Run with: python -m src.backend.tests.stress_test
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import Optional
import httpx

API_BASE = "http://localhost:8000"

# Test credentials (create this user first or use existing)
TEST_EMAIL = "stresstest@hospital.org"
TEST_PASSWORD = "StressTest123!"


@dataclass
class TestResult:
    name: str
    passed: bool
    latency_ms: float
    status_code: int
    detail: str = ""
    response_data: Optional[dict] = None


@dataclass
class StressTestReport:
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    latencies: list = field(default_factory=list)
    results: list = field(default_factory=list)

    def add(self, result: TestResult):
        self.total_tests += 1
        self.results.append(result)
        self.latencies.append(result.latency_ms)
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        if not self.latencies:
            return "No tests run"
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                    STRESS TEST REPORT                        ║
╠══════════════════════════════════════════════════════════════╣
║  Total Tests:    {self.total_tests:>4}                                       ║
║  Passed:         {self.passed:>4} ✅                                      ║
║  Failed:         {self.failed:>4} {'❌' if self.failed > 0 else '✅'}                                      ║
╠══════════════════════════════════════════════════════════════╣
║  Latency Statistics (ms):                                    ║
║    Min:          {min(self.latencies):>8.2f}                                 ║
║    Max:          {max(self.latencies):>8.2f}                                 ║
║    Mean:         {statistics.mean(self.latencies):>8.2f}                                 ║
║    Median:       {statistics.median(self.latencies):>8.2f}                                 ║
║    Std Dev:      {statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0:>8.2f}                                 ║
╚══════════════════════════════════════════════════════════════╝
"""


async def get_auth_token(client: httpx.AsyncClient) -> Optional[str]:
    """Register and/or login to get JWT token."""
    # Try to register first (may fail if user exists)
    await client.post(
        f"{API_BASE}/auth/register",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD, "role": "STAFF"},
    )

    # Now login
    response = await client.post(
        f"{API_BASE}/auth/token",
        data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
    )
    if response.status_code == 200:
        return response.json()["access_token"]
    print(f"⚠️  Auth failed: {response.status_code} - {response.text}")
    return None


# =============================================================================
# TEST SUITE 1: PARTIAL VITALS (Missing Data Handling)
# =============================================================================

PARTIAL_VITALS_CASES = [
    {
        "name": "Only Chief Complaint + Pulse",
        "payload": {
            "chief_complaint": "Chest pain radiating to left arm, shortness of breath",
            "static_vitals": {"Pulse": 110, "patient_id": 1},
        },
        "expected_behavior": "Should apply median imputation for missing SBP, RR, O2Sat, Temp",
    },
    {
        "name": "Missing Heart Rate",
        "payload": {
            "chief_complaint": "Severe headache with visual disturbances",
            "static_vitals": {"SBP": 180, "DBP": 110, "O2Sat": 97, "patient_id": 1},
        },
        "expected_behavior": "Should impute HR, still calculate triage level",
    },
    {
        "name": "Only Chief Complaint (No Vitals)",
        "payload": {
            "chief_complaint": "Feeling dizzy and nauseous after eating seafood",
            "static_vitals": {"patient_id": 1},
        },
        "expected_behavior": "Should use all median values, may produce lower acuity",
    },
    {
        "name": "Chief Complaint with Age Only",
        "payload": {
            "chief_complaint": "Fall from standing height, hip pain",
            "static_vitals": {"Age": 78, "patient_id": 1},
        },
        "expected_behavior": "Elderly fall → should still triage appropriately",
    },
    {
        "name": "Minimal Data - Walk-in",
        "payload": {
            "chief_complaint": "Sore throat for 3 days",
            "arrival_mode": "Walk-in",
            "static_vitals": {"patient_id": 1},
        },
        "expected_behavior": "Low acuity expected (KTAS 4-5)",
    },
]


async def test_partial_vitals(client: httpx.AsyncClient, token: str, report: StressTestReport):
    """Test handling of incomplete vital signs."""
    print("\n" + "=" * 60)
    print("📋 TEST SUITE 1: PARTIAL VITALS (Missing Data Handling)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {token}"}

    for case in PARTIAL_VITALS_CASES:
        start = time.perf_counter()
        try:
            response = await client.post(
                f"{API_BASE}/triage/assess",
                json=case["payload"],
                headers=headers,
                timeout=30.0,
            )
            latency = (time.perf_counter() - start) * 1000

            passed = response.status_code == 200
            detail = ""
            response_data = None

            if passed:
                response_data = response.json()
                detail = f"Level: {response_data.get('triage_level')}, Token: {response_data.get('token_number')}"
            else:
                try:
                    err_json = response.json()
                    detail = f"Error: {err_json.get('detail', response.text[:100])}"
                except:
                    detail = f"Error: {response.text[:100]}"

            result = TestResult(
                name=case["name"],
                passed=passed,
                latency_ms=latency,
                status_code=response.status_code,
                detail=detail,
                response_data=response_data,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=case["name"],
                passed=False,
                latency_ms=latency,
                status_code=0,
                detail=f"Exception: {str(e)[:100]}",
            )

        report.add(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}")
        print(f"     Expected: {case['expected_behavior']}")
        print(f"     Result: {result.detail} ({result.latency_ms:.1f}ms)")


# =============================================================================
# TEST SUITE 2: HIGH-LOAD SIMULATION (Concurrent Requests)
# =============================================================================

HIGH_LOAD_CASES = [
    {
        "chief_complaint": f"Patient #{i} - Abdominal pain, nausea",
        "static_vitals": {
            "Age": 25 + i * 5,
            "Pulse": 70 + i * 3,
            "SBP": 120 - i * 2,
            "DBP": 80,
            "O2Sat": 98,
            "Resp": 16,
            "patient_id": 1,
        },
    }
    for i in range(10)
]


async def test_high_load(client: httpx.AsyncClient, token: str, report: StressTestReport):
    """Test system under rapid concurrent requests."""
    print("\n" + "=" * 60)
    print("🔥 TEST SUITE 2: HIGH-LOAD SIMULATION (10 Concurrent Requests)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {token}"}

    async def send_request(idx: int, payload: dict) -> TestResult:
        start = time.perf_counter()
        try:
            response = await client.post(
                f"{API_BASE}/triage/assess",
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            latency = (time.perf_counter() - start) * 1000
            passed = response.status_code == 200
            detail = ""
            if passed:
                data = response.json()
                detail = f"Token: {data.get('token_number')}, Level: {data.get('triage_level')}"
            else:
                try:
                    err_json = response.json()
                    detail = f"HTTP {response.status_code}: {err_json.get('detail', '')[:40]}"
                except:
                    detail = f"HTTP {response.status_code}"
            return TestResult(
                name=f"Concurrent Request #{idx + 1}",
                passed=passed,
                latency_ms=latency,
                status_code=response.status_code,
                detail=detail,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return TestResult(
                name=f"Concurrent Request #{idx + 1}",
                passed=False,
                latency_ms=latency,
                status_code=0,
                detail=str(e)[:50],
            )

    # Fire all requests concurrently
    print("  ⏳ Sending 10 requests simultaneously...")
    overall_start = time.perf_counter()

    tasks = [send_request(i, case) for i, case in enumerate(HIGH_LOAD_CASES)]
    results = await asyncio.gather(*tasks)

    overall_time = (time.perf_counter() - overall_start) * 1000
    print(f"  ⏱️  Total wall-clock time: {overall_time:.1f}ms")

    for result in results:
        report.add(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}: {result.detail} ({result.latency_ms:.1f}ms)")

    # Throughput analysis
    successful = sum(1 for r in results if r.passed)
    print(f"\n  📊 Throughput: {successful}/10 successful")
    print(f"     Requests/sec: {10 / (overall_time / 1000):.2f}")


# =============================================================================
# TEST SUITE 3: INVALID/EXTREME INPUTS (Outlier Resilience)
# =============================================================================

INVALID_INPUT_CASES = [
    {
        "name": "HR = 0 (Zero Heart Rate)",
        "payload": {
            "chief_complaint": "Unconscious patient found",
            "static_vitals": {"Pulse": 0, "SBP": 60, "patient_id": 1},
        },
        "expected": "Should handle gracefully (impute or flag critical)",
    },
    {
        "name": "SBP = 300 (Extreme Hypertension)",
        "payload": {
            "chief_complaint": "Severe headache, vision changes",
            "static_vitals": {"Pulse": 90, "SBP": 300, "DBP": 180, "patient_id": 1},
        },
        "expected": "Should recognize hypertensive emergency",
    },
    {
        "name": "O2Sat = 50% (Severe Hypoxia)",
        "payload": {
            "chief_complaint": "Blue lips, cannot breathe",
            "static_vitals": {"Pulse": 140, "SBP": 80, "O2Sat": 50, "patient_id": 1},
        },
        "expected": "Should trigger highest acuity (KTAS 1)",
    },
    {
        "name": "Negative Values",
        "payload": {
            "chief_complaint": "Test case with impossible values",
            "static_vitals": {"Pulse": -10, "SBP": -50, "patient_id": 1},
        },
        "expected": "Should reject or sanitize negative values",
    },
    {
        "name": "Temperature = 108°F (Extreme Hyperthermia)",
        "payload": {
            "chief_complaint": "Heat stroke, altered mental status",
            "static_vitals": {"Pulse": 130, "SBP": 90, "Temp": 108, "patient_id": 1},
        },
        "expected": "Critical temperature → high acuity",
    },
    {
        "name": "Age = 150 (Impossible Age)",
        "payload": {
            "chief_complaint": "General checkup",
            "static_vitals": {"Age": 150, "Pulse": 70, "SBP": 120, "patient_id": 1},
        },
        "expected": "Should cap or handle impossible age",
    },
    {
        "name": "Empty Chief Complaint",
        "payload": {
            "chief_complaint": "",
            "static_vitals": {"Pulse": 80, "SBP": 120, "patient_id": 1},
        },
        "expected": "May fail validation or use default embedding",
    },
    {
        "name": "Very Long Chief Complaint (5000 chars)",
        "payload": {
            "chief_complaint": "Patient reports " + "pain " * 1000,
            "static_vitals": {"Pulse": 85, "SBP": 115, "patient_id": 1},
        },
        "expected": "Should truncate or handle gracefully",
    },
    {
        "name": "Shock Index > 2.0 (HR=200, SBP=80)",
        "payload": {
            "chief_complaint": "Massive hemorrhage, pale, diaphoretic",
            "static_vitals": {"Pulse": 200, "SBP": 80, "patient_id": 1},
        },
        "expected": "SI=2.5 → Critical alert, KTAS 1-2",
    },
    {
        "name": "All Zeros",
        "payload": {
            "chief_complaint": "Cardiac arrest",
            "static_vitals": {"Pulse": 0, "SBP": 0, "DBP": 0, "O2Sat": 0, "patient_id": 1},
        },
        "expected": "Immediate resuscitation level or handle as missing",
    },
]


async def test_invalid_inputs(client: httpx.AsyncClient, token: str, report: StressTestReport):
    """Test system resilience to extreme/invalid values."""
    print("\n" + "=" * 60)
    print("⚠️  TEST SUITE 3: INVALID/EXTREME INPUTS (Outlier Resilience)")
    print("=" * 60)

    headers = {"Authorization": f"Bearer {token}"}

    for case in INVALID_INPUT_CASES:
        start = time.perf_counter()
        try:
            response = await client.post(
                f"{API_BASE}/triage/assess",
                json=case["payload"],
                headers=headers,
                timeout=30.0,
            )
            latency = (time.perf_counter() - start) * 1000

            # For invalid inputs, we accept either:
            # - 200 with valid triage (system handled it)
            # - 400/422 with clear error (system rejected it)
            # We only fail if we get 500 (unhandled exception)
            passed = response.status_code != 500
            detail = ""
            response_data = None

            if response.status_code == 200:
                response_data = response.json()
                level = response_data.get("triage_level")
                token_num = response_data.get("token_number")
                detail = f"Accepted → Level: {level}, Token: {token_num}"
            elif response.status_code in (400, 422):
                detail = f"Rejected (HTTP {response.status_code})"
            else:
                detail = f"Unexpected: HTTP {response.status_code}"

            result = TestResult(
                name=case["name"],
                passed=passed,
                latency_ms=latency,
                status_code=response.status_code,
                detail=detail,
                response_data=response_data,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            result = TestResult(
                name=case["name"],
                passed=False,
                latency_ms=latency,
                status_code=0,
                detail=f"Exception: {str(e)[:80]}",
            )

        report.add(result)
        status = "✅" if result.passed else "❌"
        print(f"  {status} {result.name}")
        print(f"     Expected: {case['expected']}")
        print(f"     Result: {result.detail} ({result.latency_ms:.1f}ms)")


# =============================================================================
# TEST SUITE 4: SSE Stream Validation
# =============================================================================

async def test_sse_stream(client: httpx.AsyncClient, report: StressTestReport):
    """Test SSE dashboard stream connectivity."""
    print("\n" + "=" * 60)
    print("📡 TEST SUITE 4: SSE STREAM VALIDATION")
    print("=" * 60)

    start = time.perf_counter()
    try:
        async with client.stream("GET", f"{API_BASE}/dashboard/stream", timeout=10.0) as response:
            latency = (time.perf_counter() - start) * 1000
            
            if response.status_code == 200:
                # Try to read first few bytes to confirm stream is working
                content_type = response.headers.get("content-type", "")
                is_sse = "text/event-stream" in content_type
                
                result = TestResult(
                    name="SSE Stream Connection",
                    passed=is_sse,
                    latency_ms=latency,
                    status_code=200,
                    detail=f"Content-Type: {content_type}" if is_sse else "Not SSE stream",
                )
            else:
                result = TestResult(
                    name="SSE Stream Connection",
                    passed=False,
                    latency_ms=latency,
                    status_code=response.status_code,
                    detail=f"HTTP {response.status_code}",
                )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        result = TestResult(
            name="SSE Stream Connection",
            passed=False,
            latency_ms=latency,
            status_code=0,
            detail=f"Exception: {str(e)[:80]}",
        )

    report.add(result)
    status = "✅" if result.passed else "❌"
    print(f"  {status} {result.name}: {result.detail} ({result.latency_ms:.1f}ms)")


# =============================================================================
# MAIN RUNNER
# =============================================================================

async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🏥 TRIAGE HMS - STRESS TEST SUITE                    ║
║         Testing Robustness Against Real-World ED Chaos       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    report = StressTestReport()

    async with httpx.AsyncClient() as client:
        # Health check
        try:
            health = await client.get(f"{API_BASE}/health", timeout=5.0)
            health_data = health.json()
            
            if health.status_code != 200:
                print(f"❌ API not healthy: {health.status_code}")
                return
            
            db_status = health_data.get("database", "unknown")
            models_status = health_data.get("models", "unknown")
            dev_mode = health_data.get("dev_mode", False)
            
            print(f"✅ API Health Check Passed")
            print(f"   Database: {db_status}")
            print(f"   Models: {models_status}")
            if dev_mode:
                print(f"   ⚠️  Running in DEV MODE (mock models)\n")
            else:
                print()
                
        except Exception as e:
            print(f"❌ Cannot connect to API at {API_BASE}: {e}")
            print("   Make sure the backend is running: uvicorn src.backend.main:app")
            return

        # Get auth token
        print("🔐 Authenticating...")
        token = await get_auth_token(client)
        if not token:
            print("❌ Authentication failed. Cannot proceed with tests.")
            return
        print(f"✅ Authenticated as {TEST_EMAIL}\n")

        # Run test suites
        await test_partial_vitals(client, token, report)
        await test_high_load(client, token, report)
        await test_invalid_inputs(client, token, report)
        await test_sse_stream(client, report)

    # Print final report
    print(report.summary())

    # Detailed failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        print("\n📋 FAILED TESTS DETAIL:")
        print("-" * 60)
        for f in failures:
            print(f"  ❌ {f.name}")
            print(f"     Status: {f.status_code}")
            print(f"     Detail: {f.detail}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
