import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Eye } from "lucide-react";

interface BillingRecord {
  id: string;
  amount: number;
  date: string;
  status: "paid" | "pending" | "failed";
  planName: string;
  invoiceId: string;
}

const mockBillingHistory: BillingRecord[] = [
  {
    id: "1",
    amount: 29.99,
    date: "2024-01-15",
    status: "paid",
    planName: "Premium Plan",
    invoiceId: "INV-2024-001",
  },
  {
    id: "2",
    amount: 9.99,
    date: "2024-01-01",
    status: "paid",
    planName: "Basic Plan",
    invoiceId: "INV-2024-002",
  },
  {
    id: "3",
    amount: 99.99,
    date: "2024-02-01",
    status: "pending",
    planName: "Enterprise Plan",
    invoiceId: "INV-2024-003",
  },
  {
    id: "4",
    amount: 29.99,
    date: "2023-12-15",
    status: "paid",
    planName: "Premium Plan",
    invoiceId: "INV-2023-045",
  },
  {
    id: "5",
    amount: 9.99,
    date: "2023-11-01",
    status: "failed",
    planName: "Basic Plan",
    invoiceId: "INV-2023-032",
  },
];

export const BillingHistory = () => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "paid":
        return "bg-green-100 text-green-800 hover:bg-green-200";
      case "pending":
        return "bg-yellow-100 text-yellow-800 hover:bg-yellow-200";
      case "failed":
        return "bg-red-100 text-red-800 hover:bg-red-200";
      default:
        return "bg-gray-100 text-gray-800 hover:bg-gray-200";
    }
  };

  const getTotalPaid = () => {
    return mockBillingHistory
      .filter(record => record.status === "paid")
      .reduce((total, record) => total + record.amount, 0);
  };

  const handleDownloadInvoice = (record: BillingRecord) => {
    // Mock download functionality
    console.log("Downloading invoice:", record.invoiceId);
  };

  const handleViewInvoice = (record: BillingRecord) => {
    // Mock view functionality
    console.log("Viewing invoice:", record.invoiceId);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Billing History</h1>
        <p className="text-muted-foreground mt-2">Track your payments and download invoices</p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Paid</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">${getTotalPaid().toFixed(2)}</div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Successful Payments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {mockBillingHistory.filter(r => r.status === "paid").length}
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Pending Payments</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {mockBillingHistory.filter(r => r.status === "pending").length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Billing History Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-primary">Payment History</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[600px]">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Invoice ID</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Plan</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Amount</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Date</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Status</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {mockBillingHistory.map((record) => (
                  <tr 
                    key={record.id}
                    className="border-b border-border hover:bg-muted/50 transition-colors duration-200"
                  >
                    <td className="py-4 px-4 font-mono text-sm text-muted-foreground">
                      {record.invoiceId}
                    </td>
                    <td className="py-4 px-4 font-medium text-foreground">
                      {record.planName}
                    </td>
                    <td className="py-4 px-4 font-semibold text-foreground">
                      ${record.amount.toFixed(2)}
                    </td>
                    <td className="py-4 px-4 text-muted-foreground">
                      {record.date}
                    </td>
                    <td className="py-4 px-4">
                      <Badge className={getStatusColor(record.status)}>
                        {record.status}
                      </Badge>
                    </td>
                    <td className="py-4 px-4">
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleViewInvoice(record)}
                          className="h-8 w-8 p-0"
                        >
                          <Eye className="h-4 w-4" />
                        </Button>
                        {record.status === "paid" && (
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => handleDownloadInvoice(record)}
                            className="h-8 w-8 p-0"
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};