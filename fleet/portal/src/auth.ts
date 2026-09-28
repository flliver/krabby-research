import NextAuth from "next-auth";
import Cognito from "next-auth/providers/cognito";
import type { JWT } from "@auth/core/jwt";

declare module "next-auth" {
  interface Session {
    accessToken?: string;
  }
}

declare module "@auth/core/jwt" {
  interface JWT {
    accessToken?: string;
  }
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Cognito({
      clientId: process.env.AUTH_COGNITO_ID!,
      // Public Cognito app client (no secret); PKCE is the confidentiality mechanism.
      client: { token_endpoint_auth_method: "none" },
      issuer: process.env.AUTH_COGNITO_ISSUER!,
      checks: ["pkce", "state"],
      authorization: { params: { scope: "openid email profile" } },
    }),
  ],
  session: { strategy: "jwt" },
  callbacks: {
    authorized({ auth }) {
      return !!auth;
    },
    async jwt({ token, account }) {
      if (account?.access_token) {
        token.accessToken = account.access_token;
      }
      return token;
    },
    async session({ session, token }) {
      const jwt = token as JWT;
      if (typeof jwt.accessToken === "string") {
        session.accessToken = jwt.accessToken;
      }
      return session;
    },
  },
  pages: {
    signIn: "/login",
  },
});
