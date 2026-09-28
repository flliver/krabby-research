import { redirect } from "next/navigation";
import { auth, signIn } from "@/auth";

export default async function LoginPage() {
  const session = await auth();
  if (session) {
    redirect("/devices");
  }

  return (
    <div className="login-card">
      <h1>Sign in</h1>
      <p>Operators authenticate with Cognito. Membership in the operator group is required to view devices or open SSH tunnels.</p>
      <form
        action={async () => {
          "use server";
          await signIn("cognito", { redirectTo: "/devices" });
        }}
      >
        <button className="primary" type="submit">
          Sign in with Cognito
        </button>
      </form>
    </div>
  );
}
