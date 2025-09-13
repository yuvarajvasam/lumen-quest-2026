import { useState } from "react";
import { AdminSidebar } from "@/components/admin/AdminSidebar";
import { ManagePlans } from "@/components/admin/ManagePlans";
import { Analytics } from "@/components/admin/Analytics";
import { UsersSubscriptions } from "@/components/admin/UsersSubscriptions";
import { Discounts } from "@/components/admin/Discounts";

type AdminSection = "plans" | "analytics" | "users" | "discounts";

const Admin = () => {
  const [activeSection, setActiveSection] = useState<AdminSection>("plans");

  const renderContent = () => {
    switch (activeSection) {
      case "plans":
        return <ManagePlans />;
      case "analytics":
        return <Analytics />;
      case "users":
        return <UsersSubscriptions />;
      case "discounts":
        return <Discounts />;
      default:
        return <ManagePlans />;
    }
  };

  return (
    <div className="flex min-h-screen bg-background">
      <AdminSidebar 
        activeSection={activeSection} 
        onSectionChange={setActiveSection} 
      />
      <main className="flex-1 overflow-auto">
        <div className="container mx-auto p-6">
          {renderContent()}
        </div>
      </main>
    </div>
  );
};

export default Admin;