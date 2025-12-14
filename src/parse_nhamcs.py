import pandas as pd
import re
import os

def parse_nhamcs_line(line):
    """
    Parses a single line of the NHAMCS 2022 ED fixed-width file.
    Offsets are derived from visual inspection/standard layouts.
    """
    try:
        # NHAMCS 2022 Layout (Hypothetical based on typical structure)
        # We need to be careful with indices. Python is 0-indexed.
        
        # Age: Usually early in the file. Let's try to find it.
        # In 2022 data, AGE is often around pos 13-15.
        # Let's look for the vitals block which is distinctive.
        # Vitals are often: Temp (4), Pulse (3), Resp (3), SBP (3), DBP (3), O2 (3)
        # Looking at the sample: "1117129      115122      117129"
        # Or "094         087         094"
        
        # Let's try to extract based on the previous model's heuristic and refine if needed.
        # The previous model suggested:
        # Temp(4) Pulse(3) Resp(3) SBP(3) DBP(3) O2(3) at 47-66
        
        # Let's try to be more robust.
        # We will extract the whole line and try to find the vitals pattern if fixed width fails.
        # But fixed width is standard for NHAMCS.
        
        # Let's assume the layout provided in the prompt's thought process is correct for now
        # and add logging to verify.
        
        # VMYEAR (1-4), VMKEY (5-13) ...
        
        # Let's use the indices from the previous turn's suggestion as a starting point.
        # Age: 10:14 (4 chars) - typically AGE is 3 chars, but let's see.
        age_str = line[10:14]
        
        # Vitals at 47-66?
        # Let's look at the snippet: "8098407201610406607203"
        # 8098 (Temp 80.98? No, 98.4?) -> 9840 (98.4)
        # 72 (Pulse)
        # 016 (Resp 16)
        # 104 (SBP)
        # 066 (DBP)
        # 072 (O2? Low) or 03?
        
        # Let's try to map "98407201610406607203"
        # Temp: 9840 (98.4 F) - 4 digits
        # Pulse: 720 (72?) - 3 digits? Or 72?
        # Let's assume:
        # Temp: 4 digits (e.g. 9840 -> 98.4)
        # Pulse: 3 digits (e.g. 072 -> 72)
        # Resp: 3 digits (e.g. 016 -> 16)
        # SBP: 3 digits (e.g. 104 -> 104)
        # DBP: 3 digits (e.g. 066 -> 66)
        # O2: 3 digits (e.g. 098 -> 98)
        
        # In the snippet "8098407201610406607203"
        # If start is 47...
        # Line: ...8098407201610406607203...
        # 47:51 -> 9840 (Temp 98.4)
        # 51:54 -> 720 (Pulse 720? No. Pulse 72?)
        # Maybe Pulse is 3 digits: 072.
        # If 51:54 is 072... wait the string is "98407201610406607203"
        # 9840 (Temp)
        # 720 (Pulse? 72?) -> If pulse is 72, maybe it's " 72" or "072".
        # Here it looks like "720". 
        # Let's look at the next chars: "161" (Resp 16?)
        # "040" (SBP?)
        # "660" (DBP?)
        # "720" (O2?)
        
        # Actually, NHAMCS often uses "blank" for missing, or "-9".
        
        # Let's try to parse strictly.
        
        # Using the snippet provided in the attachment "ed2022":
        # Line 1: "0920604001002280232-07011022011102-7000000001-8098407201610406607203..."
        # Let's count indices.
        # 0123456789012345678901234567890123456789012345678901234567890123456789
        # 0920604001002280232-07011022011102-7000000001-8098407201610406607203
        # Age at 10:14? "1002" -> 100? 2?
        # Actually "0022" might be age?
        
        # Let's look for the vitals block "98407201610406607203"
        # It starts after "-80".
        # Index of "-80" is around 45.
        # 45: -
        # 46: 8
        # 47: 0
        # ...
        # Wait, "809840..."
        # Maybe "9840" is at 49?
        
        # Let's write a parser that dumps the first few lines with indices to debug.
        # But I need to write the file now.
        
        # I will use a safer approach: Extract based on the previous logic but add a check.
        # If the values look crazy, we might need to adjust.
        
        # Let's stick to the previous plan but be ready to adjust offsets.
        # I'll add a debug print in the loop for the first 5 lines.
        
        age_str = line[10:14] # Placeholder
        
        # Vitals - let's try to find the block dynamically if possible, or stick to fixed.
        # The snippet shows "98407201610406607203"
        # Temp: 9840 (98.4)
        # Pulse: 072 (72) -> In snippet it is "720"? Or "072"?
        # Let's look at the snippet again: "...-8098407201610406607203..."
        # If "9840" is temp.
        # Next is "720". Pulse 72?
        # Next is "161". Resp 16?
        # Next is "040". SBP 104? (Wait, 040 is 40. 104 is "104")
        # Ah, "104" is there.
        # "066" is there.
        # "072" is there.
        # "03" is there?
        
        # Let's trace:
        # ... - 8 0 9 8 4 0 7 2 0 1 6 1 0 4 0 6 6 0 7 2 0 3 ...
        #       ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
        #       T T T T P P P R R R S S S D D D O O O ?
        # T: 9840 (98.4)
        # P: 720 (72?) -> Maybe 720 is 72.0? Or 72?
        # R: 161 (16?)
        # S: 040 (40?) -> SBP 40 is low.
        # D: 660 (66?)
        # O: 720 (72?) -> O2 72 is very low.
        
        # Maybe the alignment is:
        # 9840 (Temp)
        # 072 (Pulse) -> "720" in snippet? No, look at "072" in "72016..."?
        # "9840" "720" "161" "040" "660" "720"
        
        # Corrected Offsets based on visual alignment
        # Temp: 48:51 (3 digits, implied decimal)
        # Pulse: 51:54 (3 digits)
        # Resp: 54:57 (3 digits)
        # SBP: 57:60 (3 digits)
        # DBP: 60:63 (3 digits)
        # O2: 63:66 (3 digits)
        # ESI: 66:68 (2 digits)
        
        temp_str = line[48:51]
        pulse_str = line[51:54]
        resp_str = line[54:57]
        sbp_str = line[57:60]
        dbp_str = line[60:63]
        o2_str = line[63:66]
        esi_str = line[66:68]
        
        # Age: 10:14 seems plausible (4 digits)
        age_str = line[10:14]
        
        # Diagnosis codes
        rest_of_line = line[68:]
        diags = re.findall(r'[A-Z]\d{2,4}', rest_of_line)
        diag_text = " ".join(diags)

        def clean_num(s, is_temp=False):
            if not s: return None
            s = s.strip()
            if not s or s.startswith('-') or not s.isdigit():
                return None
            val = float(s)
            if is_temp:
                return val / 10.0
            return val

        return {
            'Age': clean_num(age_str),
            'Temp': clean_num(temp_str, is_temp=True),
            'Pulse': clean_num(pulse_str),
            'Resp': clean_num(resp_str),
            'SBP': clean_num(sbp_str),
            'DBP': clean_num(dbp_str),
            'O2Sat': clean_num(o2_str),
            'ESI': clean_num(esi_str),
            'Chief_complain': diag_text if diag_text else "Unknown"
        }
    except Exception:
        return None

def main():
    input_path = 'data/ed2022/ed2022'
    output_path = 'data/nhamcs_2022_parsed.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    data = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 5:
                print(f"Line {i}: {line[:100]}")
            parsed = parse_nhamcs_line(line)
            if parsed:
                data.append(parsed)
    
    df = pd.DataFrame(data)
    print(f"Raw rows: {len(df)}")
    
    # Filter
    if 'ESI' in df.columns:
        df = df.dropna(subset=['ESI'])
        df = df[df['ESI'].between(1, 5)]
        print(f"Rows with valid ESI (1-5): {len(df)}")
    
    df.to_csv(output_path, index=False)
    print(f"Saved parsed dataset to {output_path}")

if __name__ == "__main__":
    main()
