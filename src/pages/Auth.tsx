import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { User, Shield, Wifi } from "lucide-react";

const Auth = () => {
  const [userCredentials, setUserCredentials] = useState({ email: "", password: "" });
  const [adminCredentials, setAdminCredentials] = useState({ email: "", password: "" });

  const handleUserLogin = () => {
    // Mock user login - redirect to user dashboard
    localStorage.setItem("userRole", "user");
    localStorage.setItem("userEmail", userCredentials.email || "user@example.com");
    window.location.href = "/user";
  };

  const handleAdminLogin = () => {
    // Mock admin login - redirect to admin dashboard
    localStorage.setItem("userRole", "admin");
    localStorage.setItem("userEmail", adminCredentials.email || "admin@lumen.com");
    window.location.href = "/admin";
  };

  const mockUsers = [
    { email: "user@example.com", password: "password", role: "user" },
    { email: "admin@lumen.com", password: "admin123", role: "admin" }
  ];

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-background via-background to-accent/5 p-4">
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center">
              <Wifi className="h-6 w-6 text-accent-foreground" />
            </div>
            <h1 className="text-3xl font-bold text-primary">LUMEN</h1>
          </div>
          <Badge className="bg-accent/10 text-accent border-accent/20 px-4 py-2">
            Quest 2.0
          </Badge>
        </div>

        <Card className="shadow-soft">
          <CardHeader className="text-center">
            <CardTitle className="text-2xl text-primary">Welcome Back</CardTitle>
            <p className="text-muted-foreground">Sign in to access your dashboard</p>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="user" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="user" className="flex items-center gap-2">
                  <User className="h-4 w-4" />
                  User
                </TabsTrigger>
                <TabsTrigger value="admin" className="flex items-center gap-2">
                  <Shield className="h-4 w-4" />
                  Admin
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="user" className="space-y-4 mt-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="user-email">Email</Label>
                    <Input
                      id="user-email"
                      type="email"
                      placeholder="user@example.com"
                      value={userCredentials.email}
                      onChange={(e) => setUserCredentials({...userCredentials, email: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="user-password">Password</Label>
                    <Input
                      id="user-password"
                      type="password"
                      placeholder="password"
                      value={userCredentials.password}
                      onChange={(e) => setUserCredentials({...userCredentials, password: e.target.value})}
                    />
                  </div>
                  <Button 
                    onClick={handleUserLogin}
                    className="w-full bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange transition-all duration-300"
                    aria-label="Sign in as user"
                  >
                    Sign In as User
                  </Button>
                </div>
              </TabsContent>
              
              <TabsContent value="admin" className="space-y-4 mt-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="admin-email">Admin Email</Label>
                    <Input
                      id="admin-email"
                      type="email"
                      placeholder="admin@lumen.com"
                      value={adminCredentials.email}
                      onChange={(e) => setAdminCredentials({...adminCredentials, email: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="admin-password">Password</Label>
                    <Input
                      id="admin-password"
                      type="password"
                      placeholder="admin123"
                      value={adminCredentials.password}
                      onChange={(e) => setAdminCredentials({...adminCredentials, password: e.target.value})}
                    />
                  </div>
                  <Button 
                    onClick={handleAdminLogin}
                    className="w-full bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange transition-all duration-300"
                    aria-label="Sign in as admin"
                  >
                    Sign In as Admin
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
            
            <div className="mt-6 pt-4 border-t border-border">
              <Button 
                onClick={() => window.location.href = "/"}
                variant="outline"
                className="w-full"
              >
                Back to Home
              </Button>
            </div>

            {/* Demo Credentials */}
            <div className="mt-6 p-4 bg-muted/50 rounded-lg">
              <h4 className="text-sm font-semibold text-foreground mb-2">Demo Credentials:</h4>
              <div className="space-y-1 text-xs text-muted-foreground">
                <p><strong>User:</strong> user@example.com / password</p>
                <p><strong>Admin:</strong> admin@lumen.com / admin123</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Auth;