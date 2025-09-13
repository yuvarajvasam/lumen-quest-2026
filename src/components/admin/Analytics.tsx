import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { TrendingUp, TrendingDown, Users, DollarSign, Calendar, Target } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from "recharts";

// Mock data for charts
const monthlyRevenue = [
  { month: "Jan", revenue: 12000, subscriptions: 120 },
  { month: "Feb", revenue: 15000, subscriptions: 150 },
  { month: "Mar", revenue: 18000, subscriptions: 180 },
  { month: "Apr", revenue: 22000, subscriptions: 220 },
  { month: "May", revenue: 25000, subscriptions: 250 },
  { month: "Jun", revenue: 28000, subscriptions: 280 },
];

const planDistribution = [
  { name: "Fibernet Basic", value: 30, color: "hsl(var(--chart-primary))" },
  { name: "Fibernet Premium", value: 35, color: "hsl(var(--chart-secondary))" },
  { name: "Fibernet Enterprise", value: 20, color: "hsl(var(--chart-tertiary))" },
  { name: "Copper Standard", value: 10, color: "hsl(var(--chart-quaternary))" },
  { name: "Copper Plus", value: 5, color: "hsl(var(--chart-quinary))" },
];

const subscriptionTrends = [
  { month: "Jan", active: 120, canceled: 8, new: 25 },
  { month: "Feb", active: 150, canceled: 12, new: 38 },
  { month: "Mar", active: 180, canceled: 15, new: 45 },
  { month: "Apr", active: 220, canceled: 18, new: 58 },
  { month: "May", active: 250, canceled: 20, new: 50 },
  { month: "Jun", active: 280, canceled: 22, new: 52 },
];

export const Analytics = () => {
  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Analytics Dashboard</h1>
          <p className="text-muted-foreground mt-1">
            Monitor subscription performance and trends
          </p>
        </div>
        <Select defaultValue="month">
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="week">Week</SelectItem>
            <SelectItem value="month">Month</SelectItem>
            <SelectItem value="year">Year</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Revenue</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">$28,000</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="mr-1 h-3 w-3 text-green-500" />
              +12% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Subscriptions</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">280</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="mr-1 h-3 w-3 text-green-500" />
              +8% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Churn Rate</CardTitle>
            <TrendingDown className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-destructive">7.8%</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingDown className="mr-1 h-3 w-3 text-red-500" />
              +2% from last month
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Revenue Per User</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">$100</div>
            <div className="flex items-center text-xs text-muted-foreground">
              <TrendingUp className="mr-1 h-3 w-3 text-green-500" />
              +5% from last month
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Monthly Revenue Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Monthly Revenue & Subscriptions</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={monthlyRevenue}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" />
                <YAxis stroke="hsl(var(--muted-foreground))" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: "hsl(var(--card))", 
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "6px"
                  }} 
                />
                <Bar dataKey="revenue" fill="hsl(var(--chart-primary))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Plan Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Plan Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={planDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {planDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Subscription Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Subscription Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={subscriptionTrends}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="month" stroke="hsl(var(--muted-foreground))" />
              <YAxis stroke="hsl(var(--muted-foreground))" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: "hsl(var(--card))", 
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "6px"
                }} 
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="active" 
                stroke="hsl(var(--chart-primary))" 
                strokeWidth={2}
                name="Active Subscriptions"
              />
              <Line 
                type="monotone" 
                dataKey="new" 
                stroke="hsl(var(--chart-secondary))" 
                strokeWidth={2}
                name="New Subscriptions"
              />
              <Line 
                type="monotone" 
                dataKey="canceled" 
                stroke="hsl(var(--destructive))" 
                strokeWidth={2}
                name="Canceled Subscriptions"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Top Performing Plans */}
      <Card>
        <CardHeader>
          <CardTitle>Plan Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              { name: "Fibernet Premium", subscribers: 98, revenue: "$4,900", growth: "+15%" },
              { name: "Fibernet Basic", subscribers: 84, revenue: "$2,520", growth: "+8%" },
              { name: "Fibernet Enterprise", subscribers: 56, revenue: "$5,600", growth: "+22%" },
              { name: "Copper Standard", subscribers: 28, revenue: "$560", growth: "+5%" },
              { name: "Copper Plus", subscribers: 14, revenue: "$490", growth: "+12%" },
            ].map((plan, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="w-2 h-8 bg-primary rounded"></div>
                  <div>
                    <p className="font-medium">{plan.name}</p>
                    <p className="text-sm text-muted-foreground">{plan.subscribers} subscribers</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-medium">{plan.revenue}</p>
                  <Badge variant="outline" className="text-green-600">
                    {plan.growth}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};