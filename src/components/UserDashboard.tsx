import { useState } from "react";
import { User, CreditCard, Package, Lightbulb, LogOut, Bell, ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { MySubscriptions } from "./dashboard/MySubscriptions";
import { BrowsePlans } from "./dashboard/BrowsePlans";
import { BillingHistory } from "./dashboard/BillingHistory";
import { Recommendations } from "./dashboard/Recommendations";
import { NotificationCenter } from "./NotificationCenter";

const UserDashboard = () => {
  const [activeSection, setActiveSection] = useState("subscriptions");
  const [isCollapsed, setIsCollapsed] = useState(false);

  const menuItems = [
    { id: "subscriptions", label: "My Subscriptions", icon: Package },
    { id: "plans", label: "Browse Plans", icon: CreditCard },
    { id: "billing", label: "Billing History", icon: User },
    { id: "recommendations", label: "Recommendations", icon: Lightbulb },
    { id: "notifications", label: "Notifications", icon: Bell },
  ];

  const handleLogout = () => {
    // Clear authentication data and redirect to auth page
    localStorage.removeItem("userRole");
    localStorage.removeItem("userEmail");
    window.location.href = "/auth";
  };

  const renderContent = () => {
    switch (activeSection) {
      case "subscriptions":
        return <MySubscriptions />;
      case "plans":
        return <BrowsePlans />;
      case "billing":
        return <BillingHistory />;
      case "recommendations":
        return <Recommendations />;
      case "notifications":
        return <NotificationCenter />;
      default:
        return <MySubscriptions />;
    }
  };

  return (
    <div className="flex min-h-screen bg-background">
      {/* Sidebar */}
      <div 
        className={cn(
          "bg-sidebar text-sidebar-foreground transition-all duration-300 flex flex-col border-r border-sidebar-border",
          isCollapsed ? "w-16" : "w-64"
        )}
      >
        {/* Header */}
        <div className="p-4 border-b border-sidebar-border">
          <div className="flex items-center justify-between">
            {!isCollapsed && (
              <h1 className="text-xl font-bold text-sidebar-foreground">Lumen Sub Sync</h1>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsCollapsed(!isCollapsed)}
              className="text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground ml-auto"
            >
              {isCollapsed ? <ChevronRight size={20} /> : <ChevronLeft size={20} />}
            </Button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {menuItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              
              return (
                <li key={item.id}>
                  <Button
                    variant="ghost"
                    onClick={() => setActiveSection(item.id)}
                    className={cn(
                      "w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors",
                      isActive && "bg-sidebar-accent text-sidebar-accent-foreground",
                      isCollapsed && "px-2"
                    )}
                  >
                    <Icon size={20} className={cn("shrink-0", !isCollapsed && "mr-3")} />
                    {!isCollapsed && <span>{item.label}</span>}
                  </Button>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Logout */}
        <div className="p-4 border-t border-sidebar-border">
          <Button
            variant="ghost"
            onClick={handleLogout}
            className={cn(
              "w-full justify-start text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors",
              isCollapsed && "px-2"
            )}
          >
            <LogOut size={20} className={cn("shrink-0", !isCollapsed && "mr-3")} />
            {!isCollapsed && <span>Logout</span>}
          </Button>
        </div>
      </div>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-6">
          {renderContent()}
        </div>
      </main>
    </div>
  );
};

export default UserDashboard;