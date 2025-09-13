import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, Zap, Target, Star } from "lucide-react";

interface Recommendation {
  id: string;
  title: string;
  description: string;
  reason: string;
  planName: string;
  currentPlan?: string;
  savings?: string;
  priority: "high" | "medium" | "low";
  icon: any;
  category: "usage" | "cost" | "performance" | "technology";
}

const mockRecommendations: Recommendation[] = [
  {
    id: "1",
    title: "Upgrade to Fibernet Premium",
    description: "You're consistently using 85% of your data quota. Upgrade to avoid throttling and get better speeds.",
    reason: "You've used 320GB out of 500GB this month (64% usage)",
    planName: "Fibernet Premium",
    currentPlan: "Fibernet Basic",
    priority: "high",
    icon: TrendingUp,
    category: "usage",
  },
  {
    id: "2",
    title: "Switch to Yearly Billing",
    description: "Save 20% by switching to annual billing for your Fibernet Premium subscription.",
    reason: "Based on your consistent usage patterns, annual billing would save you money",
    planName: "Fibernet Premium (Yearly)",
    currentPlan: "Fibernet Premium (Monthly)",
    savings: "Save $120/year",
    priority: "medium",
    icon: Target,
    category: "cost",
  },
  {
    id: "3",
    title: "Consider Fibernet Enterprise",
    description: "Your high data usage suggests you need unlimited data and faster speeds for optimal performance.",
    reason: "You're averaging 1.2TB monthly usage, exceeding most plan limits",
    planName: "Fibernet Enterprise",
    currentPlan: "Fibernet Premium",
    priority: "medium",
    icon: Zap,
    category: "performance",
  },
  {
    id: "4",
    title: "Technology Upgrade Available",
    description: "Upgrade from Copper to Fibernet for significantly better speeds and reliability.",
    reason: "Fibernet offers 10x faster speeds than your current Copper connection",
    planName: "Fibernet Basic",
    currentPlan: "Copper Plus",
    priority: "high",
    icon: Zap,
    category: "technology",
  },
  {
    id: "5",
    title: "Optimize Your Usage Pattern",
    description: "Consider scheduling large downloads during off-peak hours to better utilize your quota.",
    reason: "Peak usage detected during business hours (9 AM - 5 PM)",
    planName: "Current Plan",
    currentPlan: "Fibernet Premium",
    priority: "low",
    icon: Target,
    category: "usage",
  },
];

export const Recommendations = () => {
  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case "high":
        return "bg-red-100 text-red-800";
      case "medium":
        return "bg-yellow-100 text-yellow-800";
      case "low":
        return "bg-green-100 text-green-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "usage":
        return "bg-blue-100 text-blue-800";
      case "cost":
        return "bg-green-100 text-green-800";
      case "performance":
        return "bg-purple-100 text-purple-800";
      case "technology":
        return "bg-orange-100 text-orange-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const handleRecommendationAction = (recommendation: Recommendation) => {
    console.log("Acting on recommendation:", recommendation.title);
    // Here you would handle the recommendation action (e.g., upgrade, switch billing)
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Recommendations</h1>
        <p className="text-muted-foreground mt-2">Personalized suggestions to optimize your subscription</p>
      </div>

      {/* AI-Powered Badge */}
      <Card className="bg-gradient-to-r from-accent/10 to-primary/10 border-accent">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <Star className="h-5 w-5 text-accent" />
            <div>
              <h3 className="font-semibold text-foreground">AI-Powered Recommendations</h3>
              <p className="text-sm text-muted-foreground">
                Based on your usage patterns and subscription history
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations Grid */}
      <div className="grid gap-6 md:grid-cols-2">
        {mockRecommendations.map((recommendation) => {
          const Icon = recommendation.icon;
          return (
            <Card 
              key={recommendation.id}
              className="relative"
            >
              <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-xl">{recommendation.title}</CardTitle>
                    <p className="text-sm text-muted-foreground mt-1">
                      {recommendation.description}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Badge className={getPriorityColor(recommendation.priority)}>
                      {recommendation.priority}
                    </Badge>
                    <Badge className={getCategoryColor(recommendation.category)}>
                      {recommendation.category}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-sm font-medium">Reason:</p>
                  <p className="text-sm text-muted-foreground">{recommendation.reason}</p>
                </div>
                
                <div className="space-y-2">
                  <p className="text-sm font-medium">Plan Details:</p>
                  <div className="text-sm text-muted-foreground space-y-1">
                    {recommendation.currentPlan && (
                      <div className="flex items-center">
                        <span className="w-1.5 h-1.5 bg-muted-foreground rounded-full mr-2"></span>
                        Current: {recommendation.currentPlan}
                      </div>
                    )}
                    <div className="flex items-center">
                      <span className="w-1.5 h-1.5 bg-primary rounded-full mr-2"></span>
                      Suggested: {recommendation.planName}
                    </div>
                    {recommendation.savings && (
                      <div className="flex items-center text-green-600">
                        <span className="w-1.5 h-1.5 bg-green-600 rounded-full mr-2"></span>
                        {recommendation.savings}
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <Button
                    onClick={() => handleRecommendationAction(recommendation)}
                    className="bg-orange hover:bg-orange-hover text-orange-foreground flex-1"
                  >
                    Apply Now
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Usage Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="text-primary">Usage Insights</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-foreground">320GB</div>
              <p className="text-sm text-muted-foreground">Data Used This Month</p>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-foreground">64%</div>
              <p className="text-sm text-muted-foreground">Quota Utilization</p>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-foreground">98%</div>
              <p className="text-sm text-muted-foreground">Connection Uptime</p>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-lg">
              <div className="text-2xl font-bold text-foreground">450 Mbps</div>
              <p className="text-sm text-muted-foreground">Average Speed</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};