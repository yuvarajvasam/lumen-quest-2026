import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, CheckCircle, CreditCard, X } from "lucide-react";

interface SubscriptionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  action: {
    type: string;
    subscription: any;
  } | null;
  onConfirm: () => void;
}

export const SubscriptionModal = ({ open, onOpenChange, action, onConfirm }: SubscriptionModalProps) => {
  if (!action) return null;

  const { type, subscription } = action;

  const getModalContent = () => {
    switch (type) {
      case "subscribe":
        return {
          title: "Subscribe to Plan",
          description: "Confirm your subscription to this plan",
          icon: <CheckCircle className="h-12 w-12 text-green-600" />,
          confirmText: "Subscribe Now",
          confirmVariant: "default" as const,
        };
      case "upgrade":
        return {
          title: "Upgrade Subscription",
          description: "Upgrade to a higher tier plan",
          icon: <CheckCircle className="h-12 w-12 text-green-600" />,
          confirmText: "Upgrade Plan",
          confirmVariant: "default" as const,
        };
      case "downgrade":
        return {
          title: "Downgrade Subscription",
          description: "Downgrade to a lower tier plan",
          icon: <AlertTriangle className="h-12 w-12 text-yellow-600" />,
          confirmText: "Downgrade Plan",
          confirmVariant: "outline" as const,
        };
      case "cancel":
        return {
          title: "Cancel Subscription",
          description: "Are you sure you want to cancel this subscription?",
          icon: <X className="h-12 w-12 text-red-600" />,
          confirmText: "Cancel Subscription",
          confirmVariant: "destructive" as const,
        };
      case "renew":
        return {
          title: "Renew Subscription",
          description: "Reactivate your subscription",
          icon: <CheckCircle className="h-12 w-12 text-green-600" />,
          confirmText: "Renew Subscription",
          confirmVariant: "default" as const,
        };
      default:
        return {
          title: "Subscription Action",
          description: "Confirm this action",
          icon: <CreditCard className="h-12 w-12 text-primary" />,
          confirmText: "Confirm",
          confirmVariant: "default" as const,
        };
    }
  };

  const modalContent = getModalContent();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md border-accent shadow-orange">
        <DialogHeader className="text-center space-y-4">
          <div className="flex justify-center">
            {modalContent.icon}
          </div>
          <DialogTitle className="text-xl text-primary">{modalContent.title}</DialogTitle>
          <DialogDescription className="text-muted-foreground">
            {modalContent.description}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="bg-muted/30 p-4 rounded-lg space-y-2">
            <h4 className="font-semibold text-foreground">Plan Details</h4>
            <div className="flex items-center justify-between">
              <span className="text-muted-foreground">Plan:</span>
              <span className="font-medium text-foreground">
                {subscription.planName || subscription.name}
              </span>
            </div>
            {subscription.price && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Price:</span>
                <span className="font-semibold text-foreground">
                  ${subscription.price}
                  {subscription.type && subscription.type !== "lifetime" 
                    ? `/${subscription.type === "yearly" ? "year" : "month"}` 
                    : subscription.type === "lifetime" ? " once" : "/month"
                  }
                </span>
              </div>
            )}
            {subscription.status && (
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Current Status:</span>
                <Badge 
                  className={
                    subscription.status === "active" 
                      ? "bg-green-100 text-green-800"
                      : subscription.status === "paused"
                      ? "bg-yellow-100 text-yellow-800"
                      : "bg-red-100 text-red-800"
                  }
                >
                  {subscription.status}
                </Badge>
              </div>
            )}
          </div>

          {type === "cancel" && (
            <div className="bg-red-50 border border-red-200 p-4 rounded-lg">
              <p className="text-sm text-red-800">
                <strong>Warning:</strong> Canceling will immediately terminate your access to premium features. 
                You can reactivate your subscription at any time.
              </p>
            </div>
          )}

          {type === "subscribe" && subscription.features && (
            <div className="bg-green-50 border border-green-200 p-4 rounded-lg">
              <h5 className="font-medium text-green-800 mb-2">Included Features:</h5>
              <ul className="text-sm text-green-700 space-y-1">
                {subscription.features.slice(0, 3).map((feature: string, index: number) => (
                  <li key={index} className="flex items-center gap-2">
                    <CheckCircle className="h-3 w-3" />
                    {feature}
                  </li>
                ))}
                {subscription.features.length > 3 && (
                  <li className="text-green-600">...and {subscription.features.length - 3} more</li>
                )}
              </ul>
            </div>
          )}
        </div>

        <DialogFooter className="flex gap-2 sm:gap-2">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="flex-1"
          >
            Cancel
          </Button>
          <Button
            variant={modalContent.confirmVariant}
            onClick={onConfirm}
            className={`
              flex-1 transition-all duration-300
              ${modalContent.confirmVariant === "default" 
                ? "bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange" 
                : ""
              }
            `}
          >
            {modalContent.confirmText}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};