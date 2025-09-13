import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Shield, Zap, Users, BarChart3, Wifi, Cable } from "lucide-react";

const Index = () => {
  const features = [
    {
      icon: Shield,
      title: "Secure Management",
      description: "Role-based access control with encrypted data protection"
    },
    {
      icon: Zap,
      title: "AI Recommendations",
      description: "Personalized plan suggestions based on your usage patterns"
    },
    {
      icon: Users,
      title: "Multi-User Support",
      description: "Manage subscriptions for individuals and teams"
    },
    {
      icon: BarChart3,
      title: "Analytics Dashboard",
      description: "Track usage, trends, and optimize your subscriptions"
    }
  ];

  const broadbandProducts = [
    {
      icon: Wifi,
      name: "Fibernet",
      description: "High-speed fiber internet with unlimited data",
      speed: "Up to 1 Gbps"
    },
    {
      icon: Cable,
      name: "Broadband Copper",
      description: "Reliable copper-based internet connection",
      speed: "Up to 100 Mbps"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5">
      {/* Header */}
      <header className="border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
              <Wifi className="h-5 w-5 text-accent-foreground" />
            </div>
            <h1 className="text-2xl font-bold text-primary">LUMEN</h1>
            <span className="text-sm text-muted-foreground">Sub Sync</span>
          </div>
          <div className="flex space-x-4">
            <Button 
              onClick={() => window.location.href = "/auth"}
              className="bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange transition-all duration-300"
            >
              Get Started
            </Button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="container mx-auto px-4 py-16 text-center" aria-labelledby="hero-heading">
        <div className="max-w-4xl mx-auto space-y-8">
          <Badge className="bg-accent/10 text-accent border-accent/20 px-4 py-2">
            LUMEN Quest 2.0
          </Badge>
          <h1 id="hero-heading" className="text-5xl md:text-6xl font-bold text-foreground">
            Lumen Sub Sync
            <span className="text-accent block">Smart Subscription Management</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Streamline your broadband subscriptions with AI-powered recommendations, 
            automated management, and comprehensive analytics for both users and administrators.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button 
              onClick={() => window.location.href = "/auth"}
              size="lg"
              className="bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange transition-all duration-300"
              aria-label="Start managing your subscriptions"
            >
              Start Managing Subscriptions
              <ArrowRight className="ml-2 h-5 w-5" />
            </Button>
          </div>
        </div>
      </section>

      {/* Broadband Products */}
      <section className="container mx-auto px-4 py-16" aria-labelledby="products-heading">
        <div className="text-center mb-12">
          <h2 id="products-heading" className="text-3xl font-bold text-foreground mb-4">Our Broadband Products</h2>
          <p className="text-muted-foreground">Choose from our range of high-speed internet solutions</p>
        </div>
        <div className="grid sm:grid-cols-2 gap-6 sm:gap-8 max-w-4xl mx-auto">
          {broadbandProducts.map((product, index) => {
            const Icon = product.icon;
            return (
              <Card key={index} className="shadow-soft hover:shadow-orange transition-all duration-300 transform hover:-translate-y-1">
                <CardHeader className="text-center">
                  <div className="mx-auto w-16 h-16 bg-accent/10 rounded-full flex items-center justify-center mb-4">
                    <Icon className="h-8 w-8 text-accent" />
                  </div>
                  <CardTitle className="text-2xl text-primary">{product.name}</CardTitle>
                  <Badge className="bg-accent/10 text-accent border-accent/20 w-fit mx-auto">
                    {product.speed}
                  </Badge>
                </CardHeader>
                <CardContent className="text-center">
                  <p className="text-muted-foreground">{product.description}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* Features */}
      <section className="container mx-auto px-4 py-16" aria-labelledby="features-heading">
        <div className="text-center mb-12">
          <h2 id="features-heading" className="text-3xl font-bold text-foreground mb-4">Key Features</h2>
          <p className="text-muted-foreground">Everything you need to manage subscriptions effectively</p>
        </div>
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 sm:gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="shadow-soft hover:shadow-orange transition-all duration-300 transform hover:-translate-y-1">
                <CardHeader className="text-center">
                  <div className="mx-auto w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-accent" />
                  </div>
                  <CardTitle className="text-lg text-primary">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground text-center">{feature.description}</p>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </section>

      {/* CTA Section */}
      <section className="container mx-auto px-4 py-16">
        <Card className="bg-gradient-to-r from-accent/10 to-primary/10 border-accent shadow-soft">
          <CardContent className="text-center py-16">
            <h2 className="text-3xl font-bold text-foreground mb-4">
              Ready to Optimize Your Subscriptions?
            </h2>
            <p className="text-muted-foreground mb-8 max-w-2xl mx-auto">
              Join thousands of users who have streamlined their subscription management 
              with our AI-powered platform.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                onClick={() => window.location.href = "/auth"}
                size="lg"
                className="bg-accent hover:bg-accent-dark text-accent-foreground shadow-orange transition-all duration-300"
              >
                Get Started Now
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-background/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-8 text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-6 h-6 bg-accent rounded flex items-center justify-center">
              <Wifi className="h-4 w-4 text-accent-foreground" />
            </div>
            <span className="text-lg font-semibold text-primary">LUMEN</span>
            <span className="text-sm text-muted-foreground">Sub Sync</span>
          </div>
          <p className="text-muted-foreground">
            ©2025 Lumen Technologies. All Rights Reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
