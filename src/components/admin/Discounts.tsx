import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Plus, Edit, Trash2, Percent, Calendar } from "lucide-react";
import { DiscountModal } from "./DiscountModal";
import { DeleteConfirmModal } from "./DeleteConfirmModal";

// Mock data - will be replaced with Supabase data
const mockDiscounts = [
  {
    id: "1",
    code: "FIBER20",
    percentage: 20,
    description: "Welcome discount for new Fibernet users",
    conditions: "Valid for first-time Fibernet subscribers only",
    validFrom: "2024-01-01",
    validUntil: "2024-12-31",
    status: "Active",
    usageCount: 45,
    usageLimit: 100,
    applicablePlans: ["Fibernet Basic", "Fibernet Premium"]
  },
  {
    id: "2",
    code: "SUMMER50",
    percentage: 50,
    description: "Summer promotion for annual subscriptions",
    conditions: "Valid for annual subscriptions only",
    validFrom: "2024-06-01",
    validUntil: "2024-08-31",
    status: "Active",
    usageCount: 23,
    usageLimit: 50,
    applicablePlans: ["Fibernet Premium", "Fibernet Enterprise"]
  },
  {
    id: "3",
    code: "COPPER15",
    percentage: 15,
    description: "Copper upgrade promotion",
    conditions: "Valid for Copper plan upgrades",
    validFrom: "2024-03-01",
    validUntil: "2024-12-31",
    status: "Active",
    usageCount: 12,
    usageLimit: 75,
    applicablePlans: ["Copper Standard", "Copper Plus"]
  },
  {
    id: "4",
    code: "EXPIRED10",
    percentage: 10,
    description: "Expired promotional discount",
    conditions: "General discount",
    validFrom: "2023-01-01",
    validUntil: "2023-12-31",
    status: "Expired",
    usageCount: 150,
    usageLimit: 200,
    applicablePlans: ["Fibernet Basic"]
  },
];

export const Discounts = () => {
  const [discounts, setDiscounts] = useState(mockDiscounts);
  const [selectedDiscount, setSelectedDiscount] = useState<typeof mockDiscounts[0] | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [discountToDelete, setDiscountToDelete] = useState<string | null>(null);

  const handleCreateDiscount = () => {
    setSelectedDiscount(null);
    setIsModalOpen(true);
  };

  const handleEditDiscount = (discount: typeof mockDiscounts[0]) => {
    setSelectedDiscount(discount);
    setIsModalOpen(true);
  };

  const handleDeleteDiscount = (discountId: string) => {
    setDiscountToDelete(discountId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = () => {
    if (discountToDelete) {
      setDiscounts(discounts.filter(discount => discount.id !== discountToDelete));
      setDiscountToDelete(null);
      setIsDeleteModalOpen(false);
    }
  };

  const handleSaveDiscount = (discountData: any) => {
    if (selectedDiscount) {
      // Edit existing discount
      setDiscounts(discounts.map(discount => 
        discount.id === selectedDiscount.id 
          ? { ...discount, ...discountData }
          : discount
      ));
    } else {
      // Create new discount
      const newDiscount = {
        id: Date.now().toString(),
        usageCount: 0,
        ...discountData,
      };
      setDiscounts([...discounts, newDiscount]);
    }
    setIsModalOpen(false);
  };

  const getStatusBadge = (status: string) => {
    switch (status.toLowerCase()) {
      case "active":
        return <Badge className="bg-green-100 text-green-800">Active</Badge>;
      case "expired":
        return <Badge variant="secondary">Expired</Badge>;
      case "disabled":
        return <Badge variant="destructive">Disabled</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  const calculateUsagePercentage = (used: number, limit: number) => {
    return Math.round((used / limit) * 100);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Discount Management</h1>
          <p className="text-muted-foreground mt-1">
            Create and manage promotional discounts for your subscription plans
          </p>
        </div>
        <Button 
          onClick={handleCreateDiscount}
          className="bg-orange hover:bg-orange-hover text-orange-foreground"
        >
          <Plus size={20} className="mr-2" />
          Create Discount
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Discounts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {discounts.filter(d => d.status === "Active").length}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Usage</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {discounts.reduce((sum, d) => sum + d.usageCount, 0)}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Average Discount</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {Math.round(discounts.reduce((sum, d) => sum + d.percentage, 0) / discounts.length)}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Discounts Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {discounts.map((discount) => (
          <Card key={discount.id} className="relative">
            <CardHeader className="pb-3">
              <div className="flex justify-between items-start">
                <div>
                  <CardTitle className="text-xl flex items-center">
                    <Percent className="mr-2 h-5 w-5 text-orange" />
                    {discount.code}
                  </CardTitle>
                  <p className="text-sm text-muted-foreground mt-1">
                    {discount.description}
                  </p>
                </div>
                {getStatusBadge(discount.status)}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-3xl font-bold text-orange">
                {discount.percentage}%
                <span className="text-sm font-normal text-muted-foreground"> off</span>
              </div>
              
              <div className="space-y-2">
                <div className="text-sm">
                  <span className="font-medium">Conditions:</span>
                  <p className="text-muted-foreground">{discount.conditions}</p>
                </div>
                
                <div className="text-sm">
                  <span className="font-medium">Applicable Plans:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {discount.applicablePlans.map((plan, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {plan}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Usage:</span>
                  <span className="font-medium">
                    {discount.usageCount} / {discount.usageLimit}
                  </span>
                </div>
                <div className="w-full bg-muted rounded-full h-2">
                  <div 
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ 
                      width: `${Math.min(calculateUsagePercentage(discount.usageCount, discount.usageLimit), 100)}%` 
                    }}
                  ></div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 border-t text-sm">
                <div className="flex items-center text-muted-foreground">
                  <Calendar className="mr-1 h-3 w-3" />
                  {formatDate(discount.validFrom)}
                </div>
                <div className="flex items-center text-muted-foreground">
                  <Calendar className="mr-1 h-3 w-3" />
                  {formatDate(discount.validUntil)}
                </div>
              </div>

              <div className="flex gap-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleEditDiscount(discount)}
                  className="flex-1"
                >
                  <Edit size={16} className="mr-1" />
                  Edit
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDeleteDiscount(discount.id)}
                  className="text-destructive hover:text-destructive"
                >
                  <Trash2 size={16} />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <DiscountModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSave={handleSaveDiscount}
        discount={selectedDiscount}
      />

      <DeleteConfirmModal
        isOpen={isDeleteModalOpen}
        onClose={() => setIsDeleteModalOpen(false)}
        onConfirm={confirmDelete}
        title="Delete Discount"
        description="This action cannot be undone. This will permanently delete the discount code."
      />
    </div>
  );
};