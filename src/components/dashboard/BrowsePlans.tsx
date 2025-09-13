import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Check } from "lucide-react";
import { SubscriptionModal } from "./SubscriptionModal";

interface Plan {
  id: string;
  name: string;
  price: number;
  type: "monthly" | "yearly";
  features: string[];
  popular?: boolean;
  dataQuota: string;
  speed: string;
  technology: "fibernet" | "copper";
}

const mockPlans: Plan[] = [
  {
    id: "1",
    name: "Fibernet Basic",
    price: 29.99,
    type: "monthly",
    features: ["100GB Data Quota", "Up to 100 Mbps", "24/7 Support", "Basic Analytics"],
    dataQuota: "100GB",
    speed: "Up to 100 Mbps",
    technology: "fibernet",
  },
  {
    id: "2",
    name: "Fibernet Premium",
    price: 49.99,
    type: "monthly",
    features: ["500GB Data Quota", "Up to 500 Mbps", "Priority Support", "Advanced Analytics", "Unlimited Streaming"],
    popular: true,
    dataQuota: "500GB",
    speed: "Up to 500 Mbps",
    technology: "fibernet",
  },
  {
    id: "3",
    name: "Fibernet Enterprise",
    price: 99.99,
    type: "monthly",
    features: ["Unlimited Data", "Up to 1 Gbps", "24/7 Phone Support", "Team Collaboration", "API Access", "SLA Guarantee"],
    dataQuota: "Unlimited",
    speed: "Up to 1 Gbps",
    technology: "fibernet",
  },
  {
    id: "4",
    name: "Copper Standard",
    price: 19.99,
    type: "monthly",
    features: ["50GB Data Quota", "Up to 50 Mbps", "Email Support", "Basic Features"],
    dataQuota: "50GB",
    speed: "Up to 50 Mbps",
    technology: "copper",
  },
  {
    id: "5",
    name: "Copper Plus",
    price: 34.99,
    type: "monthly",
    features: ["200GB Data Quota", "Up to 100 Mbps", "Priority Support", "Enhanced Features"],
    dataQuota: "200GB",
    speed: "Up to 100 Mbps",
    technology: "copper",
  },
  {
    id: "6",
    name: "Fibernet Premium Yearly",
    price: 499.99,
    type: "yearly",
    features: ["500GB Data Quota", "Up to 500 Mbps", "2 Months Free", "Priority Support", "Advanced Analytics"],
    dataQuota: "500GB",
    speed: "Up to 500 Mbps",
    technology: "fibernet",
  },
];

export const BrowsePlans = () => {
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedPlan, setSelectedPlan] = useState<Plan | null>(null);

  const handleSubscribe = (plan: Plan) => {
    setSelectedPlan(plan);
    setModalOpen(true);
  };

  const confirmSubscription = () => {
    if (selectedPlan) {
      // Here you would handle the subscription logic
      console.log("Subscribing to:", selectedPlan.name);
      setModalOpen(false);
      setSelectedPlan(null);
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "monthly":
        return "bg-blue-100 text-blue-800";
      case "yearly":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getTechnologyColor = (technology: string) => {
    switch (technology) {
      case "fibernet":
        return "bg-lumen-blue-50 text-lumen-blue-600 border-lumen-blue-200";
      case "copper":
        return "bg-lumen-orange-50 text-lumen-orange-600 border-lumen-orange-200";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Browse Plans</h1>
        <p className="text-muted-foreground mt-2">Choose the perfect plan for your needs</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {mockPlans.map((plan) => (
          <Card 
            key={plan.id} 
            className={`relative ${plan.popular ? 'ring-2 ring-accent' : ''}`}
          >
            {plan.popular && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <Badge className="bg-accent text-accent-foreground px-3 py-1">
                  Most Popular
                </Badge>
              </div>
            )}
            
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-xl">{plan.name}</CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    {plan.speed}
                  </p>
                </div>
                <Badge className={getTechnologyColor(plan.technology)}>
                  {plan.technology}
                </Badge>
              </div>
              <div className="text-3xl font-bold text-primary">
                ${plan.price}
                <span className="text-sm font-normal text-muted-foreground">/month</span>
              </div>
            </CardHeader>
            
            <CardContent className="space-y-4">
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
                  <span className="text-muted-foreground">Data Quota: </span>
                  <span className="text-primary">{plan.dataQuota}</span>
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  onClick={() => handleSubscribe(plan)}
                  className={`
                    flex-1
                    ${plan.popular 
                      ? 'bg-orange hover:bg-orange-hover text-orange-foreground' 
                      : ''
                    }
                  `}
                  variant={plan.popular ? "default" : "outline"}
                >
                  Subscribe Now
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <SubscriptionModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        action={selectedPlan ? { type: "subscribe", subscription: selectedPlan as any } : null}
        onConfirm={confirmSubscription}
      />
    </div>
  );
};