import type { Metadata } from "next";
import "./globals.css";
import ShellLayout from "@/components/ShellLayout";
import { DashboardProvider } from "@/lib/DashboardContext";
import { AuthProvider } from "@/lib/AuthContext";

export const metadata: Metadata = {
  title: "TriageAI - Hospital Operating System",
  description: "AI-Powered Emergency Department Command Center",
};

// Script to prevent dark mode flash - runs before React hydrates
const darkModeScript = `
  (function() {
    try {
      var dark = localStorage.getItem('triageai-dark') === 'true';
      if (dark) {
        document.documentElement.style.colorScheme = 'dark';
        document.documentElement.classList.add('dark');
      }
    } catch (e) {}
  })();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: darkModeScript }} />
      </head>
      <body className="antialiased">
        <AuthProvider>
          <DashboardProvider>
            <ShellLayout>{children}</ShellLayout>
          </DashboardProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
