import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Triage Command Center",
  description: "AI-Powered Emergency Department Dashboard",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
