import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

interface AuthGuardProps {
  children: React.ReactNode;
  requiredRole?: "user" | "admin";
}

const AuthGuard = ({ children, requiredRole }: AuthGuardProps) => {
  const navigate = useNavigate();

  useEffect(() => {
    const userRole = localStorage.getItem("userRole");
    const userEmail = localStorage.getItem("userEmail");

    // If no authentication data exists, redirect to auth
    if (!userRole || !userEmail) {
      navigate("/auth");
      return;
    }

    // If a specific role is required, check if user has that role
    if (requiredRole && userRole !== requiredRole) {
      // If user doesn't have the required role, redirect to their appropriate dashboard
      if (userRole === "admin") {
        navigate("/admin");
      } else {
        navigate("/user");
      }
      return;
    }

    // If user is authenticated but trying to access wrong dashboard, redirect appropriately
    if (requiredRole === "admin" && userRole !== "admin") {
      navigate("/user");
      return;
    }

    if (requiredRole === "user" && userRole !== "user") {
      navigate("/admin");
      return;
    }
  }, [navigate, requiredRole]);

  // Check if user is authenticated
  const userRole = localStorage.getItem("userRole");
  const userEmail = localStorage.getItem("userEmail");

  if (!userRole || !userEmail) {
    return null; // Will redirect in useEffect
  }

  // Check role-specific access
  if (requiredRole && userRole !== requiredRole) {
    return null; // Will redirect in useEffect
  }

  return <>{children}</>;
};

export default AuthGuard;
