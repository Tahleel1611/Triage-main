import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Login - TriageAI",
  description: "Sign in to TriageAI Hospital Operating System",
};

export default function LoginLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Login page renders children directly - no ShellLayout wrapper
  // The root layout's AuthProvider is still available
  return <>{children}</>;
}
