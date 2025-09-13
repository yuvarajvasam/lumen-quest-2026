import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";

interface Discount {
  id: string;
  code: string;
  percentage: number;
  description: string;
  conditions: string;
  validFrom: string;
  validUntil: string;
  status: string;
  usageLimit: number;
  applicablePlans: string[];
}

interface DiscountModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (discountData: Partial<Discount>) => void;
  discount?: Discount | null;
}

const availablePlans = ["Basic Plan", "Pro Plan", "Enterprise Plan"];

export const DiscountModal = ({ isOpen, onClose, onSave, discount }: DiscountModalProps) => {
  const [formData, setFormData] = useState({
    code: "",
    percentage: 0,
    description: "",
    conditions: "",
    validFrom: "",
    validUntil: "",
    status: "Active",
    usageLimit: 100,
    applicablePlans: [] as string[]
  });

  useEffect(() => {
    if (discount) {
      setFormData({
        code: discount.code,
        percentage: discount.percentage,
        description: discount.description,
        conditions: discount.conditions,
        validFrom: discount.validFrom,
        validUntil: discount.validUntil,
        status: discount.status,
        usageLimit: discount.usageLimit,
        applicablePlans: discount.applicablePlans
      });
    } else {
      const today = new Date().toISOString().split('T')[0];
      const nextYear = new Date();
      nextYear.setFullYear(nextYear.getFullYear() + 1);
      const nextYearStr = nextYear.toISOString().split('T')[0];
      
      setFormData({
        code: "",
        percentage: 0,
        description: "",
        conditions: "",
        validFrom: today,
        validUntil: nextYearStr,
        status: "Active",
        usageLimit: 100,
        applicablePlans: []
      });
    }
  }, [discount, isOpen]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
    onClose();
  };

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePlanToggle = (planName: string, checked: boolean) => {
    setFormData(prev => ({
      ...prev,
      applicablePlans: checked 
        ? [...prev.applicablePlans, planName]
        : prev.applicablePlans.filter(p => p !== planName)
    }));
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {discount ? "Edit Discount" : "Create New Discount"}
          </DialogTitle>
        </DialogHeader>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="code">Discount Code</Label>
              <Input
                id="code"
                value={formData.code}
                onChange={(e) => handleInputChange("code", e.target.value.toUpperCase())}
                placeholder="WELCOME20"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="percentage">Discount Percentage</Label>
              <Input
                id="percentage"
                type="number"
                min="1"
                max="100"
                value={formData.percentage}
                onChange={(e) => handleInputChange("percentage", parseInt(e.target.value) || 0)}
                placeholder="20"
                required
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              value={formData.description}
              onChange={(e) => handleInputChange("description", e.target.value)}
              placeholder="Brief description of the discount"
              rows={2}
              required
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="conditions">Conditions</Label>
            <Textarea
              id="conditions"
              value={formData.conditions}
              onChange={(e) => handleInputChange("conditions", e.target.value)}
              placeholder="Terms and conditions for this discount"
              rows={2}
              required
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="validFrom">Valid From</Label>
              <Input
                id="validFrom"
                type="date"
                value={formData.validFrom}
                onChange={(e) => handleInputChange("validFrom", e.target.value)}
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="validUntil">Valid Until</Label>
              <Input
                id="validUntil"
                type="date"
                value={formData.validUntil}
                onChange={(e) => handleInputChange("validUntil", e.target.value)}
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="usageLimit">Usage Limit</Label>
              <Input
                id="usageLimit"
                type="number"
                min="1"
                value={formData.usageLimit}
                onChange={(e) => handleInputChange("usageLimit", parseInt(e.target.value) || 1)}
                placeholder="100"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="status">Status</Label>
              <Select value={formData.status} onValueChange={(value) => handleInputChange("status", value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="Active">Active</SelectItem>
                  <SelectItem value="Disabled">Disabled</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Applicable Plans</Label>
            <div className="space-y-2">
              {availablePlans.map((plan) => (
                <div key={plan} className="flex items-center space-x-2">
                  <Checkbox
                    id={plan}
                    checked={formData.applicablePlans.includes(plan)}
                    onCheckedChange={(checked) => handlePlanToggle(plan, checked as boolean)}
                  />
                  <Label htmlFor={plan} className="text-sm font-normal">
                    {plan}
                  </Label>
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-end space-x-2">
            <Button type="button" variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button type="submit" className="bg-orange hover:bg-orange-hover text-orange-foreground">
              {discount ? "Update Discount" : "Create Discount"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};