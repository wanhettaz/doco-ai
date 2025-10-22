# Cloud Architecture Reference

## Network Design Principles

### VPC Structure
- Use /16 CIDR blocks for main VPCs
- Subnet strategy: public, private, and database tiers
- NAT gateways in each availability zone for redundancy
- VPC peering for cross-region communication

### Security Groups vs NACLs
**Security Groups** (stateful):
- Applied at instance level
- Allow rules only (implicit deny)
- Evaluate all rules before deciding

**Network ACLs** (stateless):
- Applied at subnet level
- Allow and deny rules
- Process rules in number order

## Load Balancing Strategies

### Application Load Balancer (Layer 7)
- Content-based routing
- Host and path-based rules
- WebSocket support
- Best for HTTP/HTTPS traffic

### Network Load Balancer (Layer 4)
- Ultra-low latency
- Static IP support
- Handles millions of requests per second
- TCP/UDP traffic

## Kubernetes Considerations

### Node Architecture
- Separate node pools for different workload types
- Spot instances for batch processing
- Reserved instances for stateful services
- Cluster autoscaling based on metrics

### Storage Options
- EBS for persistent volumes (single AZ)
- EFS for multi-AZ shared storage
- S3 for object storage and backups
- Consider CSI drivers for cloud-native storage

## Observability Stack
- Prometheus for metrics collection
- Grafana for visualisation
- Loki for log aggregation
- Jaeger for distributed tracing
- OpenTelemetry for instrumentation

## Cost Optimisation
1. Right-size instances using actual usage data
2. Use spot instances for fault-tolerant workloads
3. Implement auto-scaling policies
4. Archive old data to cheaper storage tiers
5. Use reserved instances for predictable workloads