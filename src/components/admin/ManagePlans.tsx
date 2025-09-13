import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Plus, Edit, Trash2, Eye } from "lucide-react";
import { PlanModal } from "./PlanModal";
import { DeleteConfirmModal } from "./DeleteConfirmModal";

// Mock data - will be replaced with Supabase data
const mockPlans = [
  {
    id: "1",
    name: "Basic Plan",
    price: 9.99,
    autoRenewal: true,
    status: "Active",
    description: "Essential features for individuals",
    features: ["5 Projects", "10GB Storage", "Email Support"]
  },
  {
    id: "2",
    name: "Pro Plan",
    price: 29.99,
    autoRenewal: true,
    status: "Active",
    description: "Advanced features for professionals",
    features: ["Unlimited Projects", "100GB Storage", "Priority Support", "Advanced Analytics"]
  },
  {
    id: "3",
    name: "Enterprise Plan",
    price: 99.99,
    autoRenewal: false,
    status: "Inactive",
    description: "Full suite for organizations",
    features: ["Everything in Pro", "Custom Integrations", "Dedicated Support", "White Label"]
  },
];

export const ManagePlans = () => {
  const [plans, setPlans] = useState(mockPlans);
  const [selectedPlan, setSelectedPlan] = useState<typeof mockPlans[0] | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [planToDelete, setPlanToDelete] = useState<string | null>(null);

  const handleCreatePlan = () => {
    setSelectedPlan(null);
    setIsModalOpen(true);
  };

  const handleEditPlan = (plan: typeof mockPlans[0]) => {
    setSelectedPlan(plan);
    setIsModalOpen(true);
  };

  const handleDeletePlan = (planId: string) => {
    setPlanToDelete(planId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = () => {
    if (planToDelete) {
      setPlans(plans.map(plan => 
        plan.id === planToDelete 
          ? { ...plan, status: "Inactive" }
          : plan
      ));
      setPlanToDelete(null);
      setIsDeleteModalOpen(false);
    }
  };

  const handleSavePlan = (planData: any) => {
    if (selectedPlan) {
      // Edit existing plan
      setPlans(plans.map(plan => 
        plan.id === selectedPlan.id 
          ? { ...plan, ...planData }
          : plan
      ));
    } else {
      // Create new plan
      const newPlan = {
        id: Date.now().toString(),
        ...planData,
      };
      setPlans([...plans, newPlan]);
    }
    setIsModalOpen(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Manage Plans</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage subscription plans for your platform
          </p>
        </div>
        <Button 
          onClick={handleCreatePlan}
          className="bg-orange hover:bg-orange-hover text-orange-foreground"
        >
          <Plus size={20} className="mr-2" />
          Create Plan
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {plans.map((plan) => (
          <Card key={plan.id} className="relative">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-xl">{plan.name}</CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    {plan.description}
                  </p>
                </div>
                <Badge 
                  variant={plan.status === "Active" ? "default" : "secondary"}
                  className={plan.status === "Active" ? "bg-primary" : ""}
                >
                  {plan.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-3xl font-bold text-primary">
                ${plan.price}
                <span className="text-sm font-normal text-muted-foreground">/month</span>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm font-medium">Features:</p>
                <ul className="text-sm text-muted-foreground space-y-1">
                  {plan.features.map((feature, index) => (
                    <li key={index} className="flex items-center">
                      <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2"></span>
                      {feature}
                    </li>
                  ))}
                </ul>
              </div>

              <div className="flex items-center justify-between pt-4 border-t">
                <div className="text-sm">
                  <span className="text-muted-foreground">Auto-renewal: </span>
                  <span className={plan.autoRenewal ? "text-primary" : "text-muted-foreground"}>
                    {plan.autoRenewal ? "Enabled" : "Disabled"}
                  </span>
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditPlan(plan)}
                  className="flex-1"
                >
                  <Edit size={16} className="mr-1" />
                  Edit
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDeletePlan(plan.id)}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 size={16} />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <PlanModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSavePlan}
        plan={selectedPlan}
      />

      <DeleteConfirmModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onConfirm={confirmDelete}
        title="Soft Delete Plan"
        description="This will mark the plan as inactive. It will no longer be available for new subscriptions, but existing subscriptions will continue."
      />
    </div>
  );
};