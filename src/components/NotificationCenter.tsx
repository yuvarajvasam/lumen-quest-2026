import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Bell, Check, X, AlertTriangle, Info, CheckCircle } from "lucide-react";

interface Notification {
  id: string;
  title: string;
  message: string;
  type: "info" | "warning" | "success" | "error";
  timestamp: string;
  read: boolean;
  category: "subscription" | "billing" | "usage" | "system";
}

const mockNotifications: Notification[] = [
  {
    id: "1",
    title: "Data Usage Alert",
    message: "You've used 85% of your monthly data quota (425GB out of 500GB). Consider upgrading your plan.",
    type: "warning",
    timestamp: "2024-01-15T10:30:00Z",
    read: false,
    category: "usage"
  },
  {
    id: "2",
    title: "Subscription Renewed",
    message: "Your Fibernet Premium subscription has been successfully renewed for another month.",
    type: "success",
    timestamp: "2024-01-14T09:15:00Z",
    read: false,
    category: "subscription"
  },
  {
    id: "3",
    title: "Payment Processed",
    message: "Payment of $49.99 for your Fibernet Premium plan has been processed successfully.",
    type: "success",
    timestamp: "2024-01-14T09:10:00Z",
    read: true,
    category: "billing"
  },
  {
    id: "4",
    title: "New Discount Available",
    message: "SUMMER50 discount code is now available for annual subscriptions. Save 50% on your next renewal!",
    type: "info",
    timestamp: "2024-01-13T14:20:00Z",
    read: false,
    category: "system"
  },
  {
    id: "5",
    title: "Plan Upgrade Recommended",
    message: "Based on your usage patterns, we recommend upgrading to Fibernet Enterprise for unlimited data.",
    type: "info",
    timestamp: "2024-01-12T16:45:00Z",
    read: true,
    category: "subscription"
  },
  {
    id: "6",
    title: "Maintenance Scheduled",
    message: "Scheduled maintenance for Fibernet services will occur on January 20th from 2:00 AM to 4:00 AM EST.",
    type: "warning",
    timestamp: "2024-01-11T11:00:00Z",
    read: true,
    category: "system"
  }
];

export const NotificationCenter = () => {
  const [notifications, setNotifications] = useState(mockNotifications);
  const [filter, setFilter] = useState<"all" | "unread" | "subscription" | "billing" | "usage" | "system">("all");

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "success":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case "warning":
        return <AlertTriangle className="h-5 w-5 text-yellow-600" />;
      case "error":
        return <X className="h-5 w-5 text-red-600" />;
      case "info":
      default:
        return <Info className="h-5 w-5 text-blue-600" />;
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "success":
        return "border-green-200 bg-green-50";
      case "warning":
        return "border-yellow-200 bg-yellow-50";
      case "error":
        return "border-red-200 bg-red-50";
      case "info":
      default:
        return "border-blue-200 bg-blue-50";
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case "subscription":
        return "bg-purple-100 text-purple-800";
      case "billing":
        return "bg-green-100 text-green-800";
      case "usage":
        return "bg-orange-100 text-orange-800";
      case "system":
        return "bg-blue-100 text-blue-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const markAsRead = (id: string) => {
    setNotifications(notifications.map(notification => 
      notification.id === id 
        ? { ...notification, read: true }
        : notification
    ));
  };

  const markAllAsRead = () => {
    setNotifications(notifications.map(notification => 
      ({ ...notification, read: true })
    ));
  };

  const deleteNotification = (id: string) => {
    setNotifications(notifications.filter(notification => notification.id !== id));
  };

  const filteredNotifications = notifications.filter(notification => {
    if (filter === "all") return true;
    if (filter === "unread") return !notification.read;
    return notification.category === filter;
  });

  const unreadCount = notifications.filter(n => !n.read).length;

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return "Just now";
    if (diffInHours < 24) return `${diffInHours}h ago`;
    if (diffInHours < 48) return "Yesterday";
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Notifications</h1>
          <p className="text-muted-foreground mt-1">
            Stay updated with your subscription activities and system alerts
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-accent border-accent">
            {unreadCount} unread
          </Badge>
          {unreadCount > 0 && (
            <Button
              onClick={markAllAsRead}
              variant="outline"
              size="sm"
            >
              <Check className="h-4 w-4 mr-2" />
              Mark All Read
            </Button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 flex-wrap">
        {[
          { key: "all", label: "All" },
          { key: "unread", label: "Unread" },
          { key: "subscription", label: "Subscription" },
          { key: "billing", label: "Billing" },
          { key: "usage", label: "Usage" },
          { key: "system", label: "System" }
        ].map((tab) => (
          <Button
            key={tab.key}
            variant={filter === tab.key ? "default" : "outline"}
            size="sm"
            onClick={() => setFilter(tab.key as any)}
            className={filter === tab.key ? "bg-orange hover:bg-orange-hover text-orange-foreground" : ""}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Notifications List */}
      <div className="space-y-4">
        {filteredNotifications.length === 0 ? (
          <Card className="text-center py-12">
            <CardContent>
              <Bell className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">No notifications</h3>
              <p className="text-muted-foreground">
                {filter === "unread" 
                  ? "You're all caught up! No unread notifications."
                  : "No notifications found for the selected filter."
                }
              </p>
            </CardContent>
          </Card>
        ) : (
          filteredNotifications.map((notification) => (
            <Card 
              key={notification.id} 
              className={`transition-all duration-200 hover:shadow-md ${
                !notification.read ? 'border-accent bg-accent/5' : ''
              }`}
            >
              <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 mt-1">
                      {getTypeIcon(notification.type)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className={`font-semibold ${!notification.read ? 'text-foreground' : 'text-muted-foreground'}`}>
                              {notification.title}
                            </h3>
                            {!notification.read && (
                              <div className="w-2 h-2 bg-accent rounded-full"></div>
                            )}
                            <Badge className={getCategoryColor(notification.category)}>
                              {notification.category}
                            </Badge>
                          </div>
                          <p className="text-muted-foreground mb-3">
                            {notification.message}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {formatTimestamp(notification.timestamp)}
                          </p>
                        </div>
                        
                        <div className="flex gap-2">
                          {!notification.read && (
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => markAsRead(notification.id)}
                            >
                              <Check className="h-4 w-4" />
                            </Button>
                          )}
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => deleteNotification(notification.id)}
                            className="text-destructive hover:text-destructive"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};
