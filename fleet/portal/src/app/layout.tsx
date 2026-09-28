import type { ReactNode } from "react";
import { auth, signOut } from "@/auth";
import { Providers } from "@/components/Providers";
import Link from "next/link";
import "./globals.css";

export const metadata = {
  title: "Krabby Fleet",
  description: "Operator portal for the Krabby robot fleet",
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  const session = await auth();

  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Providers>
          <div className="shell">
            <header className="topbar">
              <Link className="brand" href="/devices">
                Krabby Fleet
              </Link>
              {session?.user ? (
                <form
                  action={async () => {
                    "use server";
                    await signOut({ redirectTo: "/login" });
                  }}
                >
                  <button type="submit">
                    Sign out{session.user.email ? ` (${session.user.email})` : ""}
                  </button>
                </form>
              ) : null}
            </header>
            {children}
          </div>
        </Providers>
      </body>
    </html>
  );
}
