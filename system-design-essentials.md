# System Design Essentials

A concise guide to system design fundamentals, components, and patterns for building scalable, reliable distributed systems.

---

## Related Guide

For **ML and GenAI system design** (LLM serving, **RAG** (retrieval-augmented generation) systems, agents, **MLOps** (ML operations)), see:

📖 **[ML & GenAI System Design Guide](./system-design-genai.md)** - Specialized patterns for machine learning and generative AI systems.

---

## Table of Contents

- [Core Concepts](#core-concepts)
- [Cloud Computing & Security](#cloud-computing--security)
- [Networking & VPC](#networking--vpc)
- [Key Components](#key-components)
- [Databases](#databases)
- [Caching](#caching)
- [Message Queues & Pub/Sub](#message-queues--pubsub)
- [Storage](#storage)
- [Scalability Patterns](#scalability-patterns)
- [Distributed System Patterns](#distributed-system-patterns)
- [Capacity Estimation](#capacity-estimation)
- [Common Design Examples](#common-design-examples)
- [Quick Reference](#quick-reference)
  - [Interview Checklist](#system-design-interview-checklist)
  - [Beyond Pattern Matching](#beyond-pattern-matching-the-interview-mindset)
  - [Trade-off Matrix](#trade-off-decision-matrix)

---

## Core Concepts

### ACID Properties

**ACID** (Atomicity, Consistency, Isolation, Durability) describes four fundamental properties database transactions must satisfy to ensure data integrity, especially in systems handling financial transactions, inventory management, or any scenario where partial updates could lead to inconsistent states.

**Atomicity** ensures that a transaction is treated as a single, indivisible unit. Consider a bank transfer: if money is debited from Account A but the credit to Account B fails, atomicity guarantees the entire transaction rolls back—you'll never have money disappear into thin air.

**Consistency** guarantees that a transaction brings the database from one valid state to another. If your business rule says account balances can't be negative, the database will reject any transaction that would violate this constraint.

**Isolation** prevents concurrent transactions from interfering with each other. When two users try to book the last seat on a flight simultaneously, isolation ensures only one succeeds while the other receives an appropriate error.

**Durability** promises that once a transaction commits, it stays committed—even if the server crashes immediately after. This is typically achieved through write-ahead logging and redundant storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                        ACID PROPERTIES                          │
├────────────────┬────────────────────────────────────────────────┤
│   ATOMICITY    │  All-or-nothing: entire transaction succeeds   │
│                │  or entire transaction rolls back              │
├────────────────┼────────────────────────────────────────────────┤
│  CONSISTENCY   │  Data always valid according to all rules      │
│                │  and constraints                               │
├────────────────┼────────────────────────────────────────────────┤
│   ISOLATION    │  Concurrent transactions don't interfere       │
│                │  (appear sequential)                           │
├────────────────┼────────────────────────────────────────────────┤
│   DURABILITY   │  Committed data persists even after            │
│                │  system failure                                │
└────────────────┴────────────────────────────────────────────────┘
```

### CAP Theorem

The CAP theorem (Brewer's theorem) states that a distributed system can only guarantee two of three properties simultaneously: **Consistency**, **Availability**, and **Partition Tolerance**. Since network partitions are inevitable in distributed systems, you're essentially choosing between consistency and availability.

**Why can't we have all three?** Imagine two database nodes that lose network connectivity (a partition). When a write comes in, you have two choices:
1. **Accept the write** on the available node (choose Availability) → but now the nodes have different data (sacrifice Consistency)
2. **Reject the write** until nodes can sync (choose Consistency) → but now the system is unavailable (sacrifice Availability)

**CP systems** (like HBase, MongoDB in certain configurations) will refuse to serve requests during network issues to maintain consistency. Use these when correctness matters more than uptime—financial systems, inventory tracking, or coordination services.

**AP systems** (like Cassandra, DynamoDB) remain available but may return stale data. Use these when uptime is critical and eventual consistency is acceptable—social media feeds, shopping carts, or analytics.

```
                      CONSISTENCY
                          ╱╲
                         ╱  ╲
                        ╱    ╲
                       ╱  CP  ╲
                      ╱        ╲
                     ╱──────────╲
                    ╱            ╲
                   ╱      CA      ╲
                  ╱   (impossible  ╲
                 ╱   in distributed)╲
                ╱                    ╲
               ╱────────────────────╲
              ╱          AP          ╲
             ╱                        ╲
            ╱                          ╲
    AVAILABILITY ──────────────────── PARTITION
                                      TOLERANCE

    CP: HBase, MongoDB, Redis         AP: Cassandra, DynamoDB, CouchDB
```

### Reliability, Scalability, Maintainability

These three qualities define whether a system will succeed in production. A system that's fast but crashes constantly is useless. A system that's reliable but can't handle growth will eventually fail. A system that works but nobody can understand or modify will become a liability.

**Reliability** means the system continues functioning correctly even when things go wrong. Hardware fails, software has bugs, and humans make mistakes. Netflix's Chaos Monkey deliberately kills production servers to ensure the system can handle failures gracefully. Key techniques include redundancy, monitoring, graceful degradation, and comprehensive testing.

**Scalability** is the system's ability to handle increased load. This could mean more users, more data, or more complex operations. The two approaches are vertical scaling (bigger machines) and horizontal scaling (more machines). Most modern systems prefer horizontal scaling because it has no theoretical limit and provides better fault tolerance.

**Maintainability** determines whether your team can effectively operate, understand, and evolve the system. Good operability means easy deployment, monitoring, and debugging. Simplicity means new engineers can understand the system quickly. Evolvability means you can adapt to changing requirements without a complete rewrite.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM QUALITIES                             │
├─────────────────┬───────────────────────────────────────────────┤
│   RELIABILITY   │  • Functions correctly despite faults         │
│                 │  • Hardware, software, human error tolerance  │
│                 │  • Techniques: Chaos Monkey, monitoring       │
├─────────────────┼───────────────────────────────────────────────┤
│   SCALABILITY   │  • Handles growth without degradation         │
│                 │  • Vertical (scale up) or Horizontal (out)    │
│                 │  • Auto-scaling, load balancing               │
├─────────────────┼───────────────────────────────────────────────┤
│ MAINTAINABILITY │  • Operability: Easy to run                   │
│                 │  • Simplicity: Easy to understand             │
│                 │  • Evolvability: Easy to change               │
└─────────────────┴───────────────────────────────────────────────┘
```

### Concurrency Control

When multiple transactions access shared data simultaneously, we need mechanisms to ensure correctness. Without proper concurrency control, you could end up with lost updates, dirty reads, or phantom reads.

**Two-Phase Commit (2PC)** is a distributed transaction protocol that ensures all participants in a transaction either commit or abort together. It works in two phases:

1. **Prepare Phase**: The coordinator asks all participants "Can you commit?" Each participant prepares the transaction (acquires locks, writes to log) and votes YES or NO.
2. **Commit Phase**: If all vote YES, the coordinator tells everyone to commit. If any vote NO, everyone aborts.

The limitation of 2PC is that it's blocking—if the coordinator fails after sending PREPARE but before sending COMMIT, participants are stuck waiting. This is why many modern systems prefer eventual consistency patterns.

```
┌─────────────────────────────────────────────────────────────────┐
│                 TWO-PHASE COMMIT (2PC)                          │
│                                                                 │
│   Coordinator                Participants                       │
│       │                      │         │                        │
│       │──── PREPARE ────────►│         │                        │
│       │──── PREPARE ──────────────────►│                        │
│       │                      │         │                        │
│       │◄─── VOTE YES ────────│         │                        │
│       │◄─── VOTE YES ──────────────────│                        │
│       │                      │         │                        │
│       │──── COMMIT ─────────►│         │                        │
│       │──── COMMIT ──────────────────►│                        │
│       │                      │         │                        │
│       │◄─── ACK ─────────────│         │                        │
│       │◄─── ACK ────────────────────────│                        │
└─────────────────────────────────────────────────────────────────┘
```

**SAGA Pattern** is an alternative for long-running distributed transactions. Instead of locking resources across multiple services, a saga breaks the transaction into a sequence of local transactions, each with a compensating action that can undo its effects.

For example, in an e-commerce order:
- T1: Reserve inventory → C1: Release inventory
- T2: Charge payment → C2: Refund payment  
- T3: Ship order → C3: Cancel shipment

If T3 fails, the saga executes C2, then C1, rolling back the entire business transaction without distributed locks.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAGA PATTERN                               │
│                                                                 │
│   T1 ──► T2 ──► T3 ──► T4 ──► SUCCESS                         │
│                  │                                              │
│                  ▼ (failure)                                    │
│                 C3 ◄── C2 ◄── C1  (compensating transactions)  │
│                                                                 │
│   Each step has a compensating action for rollback              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cloud Computing & Security

### Cloud Computing Overview

Cloud computing fundamentally changed how we build and deploy applications. Instead of purchasing physical servers, estimating capacity years in advance, and managing data centers, you can provision resources on-demand and pay only for what you use.

The three main service categories are:
- **Compute**: Virtual servers (EC2), containers (ECS/EKS), or serverless functions (Lambda)
- **Storage**: Object storage (S3), block storage (EBS), or file systems (EFS)
- **Database**: Managed relational databases (RDS), NoSQL (DynamoDB), or caching (ElastiCache)

All these services communicate through a **networking layer** (**VPC**, virtual private cloud) that you configure with your own IP ranges, subnets, and routing rules. This gives you the flexibility of the cloud with the isolation of a private data center.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLOUD COMPUTING                              │
│                                                                 │
│   On-demand delivery of IT resources via internet               │
│   with pay-as-you-go pricing                                    │
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │   COMPUTE   │  │   STORAGE   │  │  DATABASE   │            │
│   │  EC2/Lambda │  │  S3/EBS/EFS │  │  RDS/Dynamo │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│          │                │                │                    │
│          └────────────────┼────────────────┘                    │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   NETWORKING    │                            │
│                  │   VPC/Route53   │                            │
│                  └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Security Fundamentals (Defense in Depth)

Security should never be an afterthought—it must be built into the architecture from day one. The **Defense in Depth** approach layers multiple security controls so that if one fails, others still protect your assets.

Cloud security rests on three fundamental pillars:

**IAM (Identity and Access Management)** controls WHO can do WHAT. Create users for individuals, groups for teams, and roles for services. Every permission should follow the principle of least privilege—grant only the minimum access needed for the task. Enable MFA for all human users.

**Encryption** protects data both at rest and in transit. Use KMS (Key Management Service) to encrypt data stored in databases and S3. Use TLS/SSL for all network communication. For the most sensitive workloads, consider HSM (Hardware Security Modules) for key storage.

**Network Security** controls traffic flow. VPCs isolate your resources, subnets segment your network into public and private zones, Security Groups act as instance-level firewalls, and NACLs provide subnet-level filtering.

```
┌─────────────────────────────────────────────────────────────────┐
│              CLOUD SECURITY - THREE PILLARS                     │
│                                                                 │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│    │     IAM      │  │  ENCRYPTION  │  │   NETWORK    │        │
│    │              │  │              │  │   SECURITY   │        │
│    ├──────────────┤  ├──────────────┤  ├──────────────┤        │
│    │ • Users      │  │ • At Rest    │  │ • VPC        │        │
│    │ • Groups     │  │   (KMS)      │  │ • Subnets    │        │
│    │ • Roles      │  │ • In Transit │  │ • Security   │        │
│    │ • Policies   │  │   (TLS/SSL)  │  │   Groups     │        │
│    │ • MFA        │  │ • HSM        │  │ • NACLs      │        │
│    └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                 │
│              PRINCIPLE: LEAST PRIVILEGE                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3-Tier Application Security

A typical web application has three tiers, each with different security requirements. The key insight is that each tier should only communicate with its adjacent layers—the internet talks to the web tier, the web tier talks to the app tier, and the app tier talks to the database. No direct internet access to your database!

**Web Tier** sits in a public subnet and handles incoming HTTP/HTTPS traffic. Its security group allows inbound traffic on ports 80 and 443 from anywhere, but outbound traffic only goes to the application tier. All traffic should use TLS/SSL encryption.

**Application Tier** resides in a private subnet with no direct internet access. It accepts traffic only from the web tier's security group and can initiate connections to the database tier. Administrative access (SSH/RDP) should go through a bastion host or VPN, never directly from the internet.

**Database Tier** is the most protected layer, also in a private subnet. It accepts connections only from the application tier on the database port (e.g., 3306 for MySQL). Enable encryption at rest using KMS and encryption in transit using SSL certificates. Never expose database ports to the internet.

```
┌─────────────────────────────────────────────────────────────────┐
│                         INTERNET                                │
│                            │                                    │
│                      ┌─────┴─────┐                              │
│                      │ FIREWALL  │                              │
│                      └─────┬─────┘                              │
│  ════════════════════════════════════════════════════════════   │
│                        PUBLIC SUBNET                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    WEB TIER (EC2)                        │   │
│  │                                                          │   │
│  │   Security Group: Inbound 80/443 from Internet          │   │
│  │                   Outbound to App Tier                   │   │
│  │   Protocol: HTTPS (TLS/SSL)                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│  ════════════════════════════════════════════════════════════   │
│                       PRIVATE SUBNET                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   APP TIER (EC2)                         │   │
│  │                                                          │   │
│  │   Security Group: Inbound from Web Tier only            │   │
│  │                   SSH/RDP for admin (via bastion)        │   │
│  │                   Outbound to DB Tier                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│  ════════════════════════════════════════════════════════════   │
│                       PRIVATE SUBNET                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  DATABASE TIER (RDS)                     │   │
│  │                                                          │   │
│  │   Security Group: Inbound from App Tier only            │   │
│  │                   No internet access                     │   │
│  │   Encryption: At rest (KMS) + In transit (SSL)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Security Groups vs NACLs

Both Security Groups and Network ACLs (NACLs) filter traffic, but they operate at different levels and have important behavioral differences.

**Security Groups** are your first line of defense, operating at the instance level. They're **stateful**, meaning if you allow inbound traffic, the response is automatically allowed outbound—you don't need separate rules for request and response. Security Groups only have ALLOW rules; anything not explicitly allowed is denied.

**NACLs** operate at the subnet level and serve as a second layer of defense. They're **stateless**, so you must explicitly allow both inbound requests AND outbound responses. NACLs support both ALLOW and DENY rules, processed in numerical order—the first matching rule wins.

**When to use each**: Use Security Groups for most filtering needs since they're easier to manage. Use NACLs when you need to explicitly deny specific IP addresses or ranges, or when you want subnet-wide rules that apply regardless of instance security groups.

```
┌─────────────────────────────────────────────────────────────────┐
│         SECURITY GROUPS              vs           NACLs         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────┐              ┌───────────────────┐      │
│  │  Instance Level   │              │   Subnet Level    │      │
│  │  (1st defense)    │              │   (2nd defense)   │      │
│  └───────────────────┘              └───────────────────┘      │
│                                                                 │
│  • STATEFUL                         • STATELESS                │
│    (return traffic                    (must explicitly         │
│     auto-allowed)                      allow return)           │
│                                                                 │
│  • Allow rules only                 • Allow AND Deny rules     │
│                                                                 │
│  • All rules evaluated              • Rules processed in       │
│                                       order (numbered)          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                       VPC                                │   │
│  │  ┌────────────────────────────────────────────────┐     │   │
│  │  │              SUBNET (NACL)                      │     │   │
│  │  │  ┌──────────────┐    ┌──────────────┐          │     │   │
│  │  │  │   EC2 (SG)   │    │   EC2 (SG)   │          │     │   │
│  │  │  └──────────────┘    └──────────────┘          │     │   │
│  │  └────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Networking & VPC

### VPC Architecture

A Virtual Private Cloud (VPC) is your isolated network within the cloud. Think of it as your own private data center, but without the physical hardware to manage. When you create a VPC, you define a CIDR block (e.g., 172.31.0.0/16) that determines the IP address range for all resources within it.

**Subnets** divide your VPC into smaller network segments, each residing in a single Availability Zone. Public subnets have routes to an Internet Gateway, allowing resources to communicate with the internet. Private subnets don't have direct internet access—resources here can only reach the internet through a NAT Gateway (for outbound-only traffic).

**Route Tables** determine where network traffic is directed. Each subnet associates with a route table that contains rules like "send 0.0.0.0/0 (all internet traffic) to the Internet Gateway" or "send traffic to other subnets via the local router."

**Internet Gateway** enables communication between your VPC and the internet. It's horizontally scaled, redundant, and highly available by default.

**NAT Gateway** allows private subnet resources to initiate outbound internet connections (e.g., to download software updates) while preventing unsolicited inbound connections. The internet can only respond to requests—it cannot initiate connections to your private resources.

```
┌─────────────────────────────────────────────────────────────────┐
│                      AWS REGION                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                    VPC (172.31.0.0/16)                     │ │
│  │                                                            │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐       │ │
│  │  │   Availability      │    │   Availability      │       │ │
│  │  │      Zone A         │    │      Zone B         │       │ │
│  │  │                     │    │                     │       │ │
│  │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │       │ │
│  │  │ │ Public Subnet   │ │    │ │ Public Subnet   │ │       │ │
│  │  │ │ 172.31.1.0/24   │ │    │ │ 172.31.3.0/24   │ │       │ │
│  │  │ │  ┌───┐  ┌───┐   │ │    │ │  ┌───┐  ┌───┐   │ │       │ │
│  │  │ │  │EC2│  │NAT│   │ │    │ │  │EC2│  │NAT│   │ │       │ │
│  │  │ │  └───┘  └───┘   │ │    │ │  └───┘  └───┘   │ │       │ │
│  │  │ └─────────────────┘ │    │ └─────────────────┘ │       │ │
│  │  │                     │    │                     │       │ │
│  │  │ ┌─────────────────┐ │    │ ┌─────────────────┐ │       │ │
│  │  │ │ Private Subnet  │ │    │ │ Private Subnet  │ │       │ │
│  │  │ │ 172.31.2.0/24   │ │    │ │ 172.31.4.0/24   │ │       │ │
│  │  │ │  ┌───┐  ┌───┐   │ │    │ │  ┌───┐  ┌───┐   │ │       │ │
│  │  │ │  │EC2│  │RDS│   │ │    │ │  │EC2│  │RDS│   │ │       │ │
│  │  │ │  └───┘  └───┘   │ │    │ │  └───┘  └───┘   │ │       │ │
│  │  │ └─────────────────┘ │    │ └─────────────────┘ │       │ │
│  │  └─────────────────────┘    └─────────────────────┘       │ │
│  │              │                         │                   │ │
│  │              └───────────┬─────────────┘                   │ │
│  │                    ┌─────┴─────┐                           │ │
│  │                    │  Router   │                           │ │
│  │                    │  (Route   │                           │ │
│  │                    │   Table)  │                           │ │
│  │                    └─────┬─────┘                           │ │
│  └──────────────────────────│────────────────────────────────┘ │
│                             │                                   │
│                    ┌────────┴────────┐                         │
│                    │ Internet Gateway │                         │
│                    └────────┬────────┘                         │
└─────────────────────────────│───────────────────────────────────┘
                              │
                         INTERNET
```

### DNS Resolution Flow

**DNS** (Domain Name System) translates domain names to IP addresses. When you type a URL into your browser, a chain of queries performs that translation. This process typically takes milliseconds but involves multiple servers across the internet.

**Step 1-2**: Your browser first checks its cache, then asks your configured DNS resolver (often your ISP's server or a public resolver like 8.8.8.8). If the resolver doesn't have the answer cached, it begins a recursive lookup.

**Step 3**: The resolver queries a root server. Root servers don't know specific domains, but they know which servers handle top-level domains like .com, .org, or .io. There are only 13 root server addresses (though many physical servers behind them).

**Step 4**: The resolver queries the TLD server for .com (or whatever the domain's TLD is). The TLD server responds with the authoritative nameservers for the specific domain.

**Step 5-6**: Finally, the resolver queries the authoritative nameserver for example.com, which returns the actual IP address. This answer is cached at various levels with TTL (time-to-live) values.

**Step 7**: Your browser connects to the web server at the returned IP address.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DNS RESOLUTION                               │
│                                                                 │
│   User types: www.example.com                                   │
│                                                                 │
│   ┌──────┐     ┌──────────┐     ┌──────────┐                   │
│   │Client│────►│   DNS    │────►│   Root   │                   │
│   │      │  1  │ Resolver │  2  │  Server  │                   │
│   └──────┘     └──────────┘     └────┬─────┘                   │
│                     │                 │                         │
│                     │    ┌────────────┘                         │
│                     │    │ 3 "Go to .com TLD"                   │
│                     │    ▼                                      │
│                     │  ┌──────────┐                             │
│                     │  │   TLD    │                             │
│                     │  │  Server  │                             │
│                     │  │  (.com)  │                             │
│                     │  └────┬─────┘                             │
│                     │       │ 4 "Go to example.com NS"          │
│                     │       ▼                                   │
│                     │  ┌──────────────┐                         │
│                     │  │Authoritative │                         │
│                     │  │    Server    │                         │
│                     │  │(example.com) │                         │
│                     │  └──────┬───────┘                         │
│                     │         │ 5 IP: 93.184.216.34             │
│                     │◄────────┘                                 │
│                     │                                           │
│   ┌──────┐◄─────────┘ 6 Return IP to client                    │
│   │Client│                                                      │
│   └──┬───┘                                                      │
│      │ 7 Connect to web server                                  │
│      ▼                                                          │
│   ┌──────────┐                                                  │
│   │Web Server│                                                  │
│   │93.184... │                                                  │
│   └──────────┘                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### TCP/IP Model

The TCP/IP model describes how data travels across networks in layers, with each layer handling specific responsibilities. Understanding this model helps you troubleshoot network issues and design systems that communicate efficiently.

**Physical Layer** deals with the actual hardware—cables, network interface cards, and the electrical or optical signals that carry data.

**Data Link Layer** handles communication between directly connected nodes. Ethernet is the most common protocol here. Switches operate at this layer, forwarding frames based on MAC addresses.

**Network Layer** routes packets across different networks. IP (Internet Protocol) operates here, and routers make forwarding decisions based on IP addresses. This is where logical addressing happens.

**Transport Layer** ensures reliable (**TCP**, Transmission Control Protocol) or fast (**UDP**, User Datagram Protocol) delivery between applications. TCP breaks data into segments, handles acknowledgments, retransmissions, and flow control. UDP is simpler and faster but doesn't guarantee delivery—perfect for real-time applications like video calls or gaming where a dropped packet matters less than latency.

**Application Layer** is where HTTP, FTP, SMTP, and other protocols live. This is the interface between network communication and your application code.

```
┌─────────────────────────────────────────────────────────────────┐
│                     TCP/IP MODEL                                │
│                                                                 │
│  Layer 5  ┌─────────────────────────────────────────────────┐  │
│           │              APPLICATION                         │  │
│           │         HTTP, FTP, SMTP, DNS                     │  │
│           └─────────────────────────────────────────────────┘  │
│                              │                                  │
│  Layer 4  ┌─────────────────────────────────────────────────┐  │
│           │               TRANSPORT                          │  │
│           │           TCP (reliable)                         │  │
│           │           UDP (fast)                             │  │
│           └─────────────────────────────────────────────────┘  │
│                              │                                  │
│  Layer 3  ┌─────────────────────────────────────────────────┐  │
│           │                NETWORK                           │  │
│           │            IP, Routers                           │  │
│           └─────────────────────────────────────────────────┘  │
│                              │                                  │
│  Layer 2  ┌─────────────────────────────────────────────────┐  │
│           │               DATA LINK                          │  │
│           │          Ethernet, Switches                      │  │
│           └─────────────────────────────────────────────────┘  │
│                              │                                  │
│  Layer 1  ┌─────────────────────────────────────────────────┐  │
│           │               PHYSICAL                           │  │
│           │         Cables, NICs, Hubs                       │  │
│           └─────────────────────────────────────────────────┘  │
│                                                                 │
│  TCP vs UDP:                                                   │
│  • TCP: Connection-oriented, reliable (file transfer, web)     │
│  • UDP: Connectionless, fast (streaming, gaming, VoIP)         │
└─────────────────────────────────────────────────────────────────┘
```

### Proxies

Proxies are intermediaries that sit between clients and servers, providing various benefits like security, caching, and load distribution. The key distinction is which side of the connection they represent.

**Forward Proxy** acts on behalf of clients. When you configure your browser to use a corporate proxy, all your requests go through it first. The destination server sees the proxy's IP, not yours. Use cases include content filtering (blocking certain websites), anonymous browsing, bypassing geo-restrictions, and caching frequently accessed content.

**Reverse Proxy** acts on behalf of servers—clients don't even know it exists. When you visit a website, you might actually connect to a reverse proxy that then forwards your request to one of many backend servers. Common reverse proxies include Nginx, HAProxy, and AWS ALB. Use cases include load balancing (distributing traffic across servers), SSL termination (handling encryption centrally), caching (reducing load on backend servers), and security (hiding backend infrastructure and providing WAF capabilities).

```
┌─────────────────────────────────────────────────────────────────┐
│                      PROXY TYPES                                │
│                                                                 │
│   FORWARD PROXY                    REVERSE PROXY                │
│   (Client-side)                    (Server-side)                │
│                                                                 │
│   ┌──────┐   ┌───────┐            ┌───────┐   ┌──────┐         │
│   │Client│──►│Forward│──►         │Reverse│◄──│Server│         │
│   └──────┘   │ Proxy │   Internet │ Proxy │   └──────┘         │
│              └───────┘            └───────┘                     │
│                  │                    │                         │
│                  ▼                    ▼                         │
│   • Masks client IP           • Load balancing                 │
│   • Content filtering         • SSL termination                │
│   • Caching                   • Caching                        │
│   • Access control            • Security (WAF)                 │
│                                                                 │
│   Example: Corporate proxy    Example: Nginx, HAProxy          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### Load Balancer with Auto Scaling

Load balancers distribute incoming traffic across multiple servers, preventing any single server from becoming overwhelmed and enabling horizontal scaling. Combined with auto-scaling, your application can automatically adjust capacity based on demand.

**How it works**: Clients connect to the load balancer's address (often via a DNS name like api.example.com). The load balancer monitors the health of backend servers and routes requests only to healthy instances. If a server fails health checks, traffic automatically shifts to remaining healthy servers.

**Auto Scaling Groups** manage the lifecycle of your servers. You define minimum (always running), maximum (cost cap), and desired capacity (normal state). Scaling policies adjust capacity based on metrics like CPU utilization, request count, or custom CloudWatch metrics. For predictable traffic patterns (like a daily sales event), scheduled scaling can pre-warm capacity before the spike hits.

**Key tip for high-traffic events**: Don't rely solely on reactive scaling. Pre-warm your load balancer and use scheduled scaling to have instances ready before the traffic arrives. The startup time for new instances includes launching the VM, running bootstrap scripts, and warming application caches—often several minutes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTO SCALING GROUP                           │
│                                                                 │
│                        INTERNET                                 │
│                           │                                     │
│                    ┌──────┴──────┐                              │
│                    │    Route    │                              │
│                    │     53      │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│                    ┌──────┴──────┐                              │
│                    │    Load     │                              │
│                    │  Balancer   │                              │
│                    │  (ALB/NLB)  │                              │
│                    └──────┬──────┘                              │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐               │
│    │   EC2   │      │   EC2   │      │   EC2   │               │
│    │Instance │      │Instance │      │Instance │               │
│    │   #1    │      │   #2    │      │   #3    │               │
│    └─────────┘      └─────────┘      └─────────┘               │
│         │                 │                 │                   │
│         └─────────────────┼─────────────────┘                   │
│                           │                                     │
│    ┌──────────────────────┴──────────────────────┐             │
│    │            AUTO SCALING POLICIES             │             │
│    │                                              │             │
│    │  • Min: 2 instances    • Max: 10 instances  │             │
│    │  • Scale out: CPU > 70%                     │             │
│    │  • Scale in:  CPU < 30%                     │             │
│    │  • Scheduled scaling for peak events        │             │
│    └──────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Load Balancer Types

The choice between Layer 4 and Layer 7 load balancers depends on your requirements for performance versus intelligence.

**Layer 4 (Network Load Balancer)** operates at the transport layer, making routing decisions based only on IP addresses and TCP/UDP ports. It's extremely fast—millions of requests per second with ultra-low latency—because it doesn't inspect packet contents. Use NLB when you need raw performance, non-HTTP protocols (gaming, IoT), or need to preserve client IPs.

**Layer 7 (Application Load Balancer)** operates at the HTTP/HTTPS layer, understanding the content of requests. It can route based on URL paths (/api/* to API servers, /images/* to media servers), HTTP headers, cookies, or even query strings. It handles SSL termination, freeing your servers from encryption overhead. Use ALB when you need content-based routing, sticky sessions, or WebSocket support.

**Load balancing algorithms** determine how traffic is distributed:
- **Round Robin**: Requests rotate through servers sequentially. Simple but doesn't account for server capacity differences.
- **Least Connections**: Sends traffic to the server with fewest active connections. Better for varying request durations.
- **IP Hash**: Uses client IP to determine the server, ensuring a client always hits the same server (sticky sessions without cookies).

```
┌─────────────────────────────────────────────────────────────────┐
│                   LOAD BALANCER TYPES                           │
│                                                                 │
│   LAYER 4 (Transport)          LAYER 7 (Application)           │
│   ┌─────────────────┐          ┌─────────────────┐             │
│   │ Network Load    │          │ Application     │             │
│   │   Balancer      │          │ Load Balancer   │             │
│   └────────┬────────┘          └────────┬────────┘             │
│            │                            │                       │
│   • TCP/UDP routing            • HTTP/HTTPS routing            │
│   • IP + Port based            • URL, Headers, Cookies         │
│   • Very fast (millions RPS)   • Content-based routing         │
│   • No inspection              • TLS termination               │
│   • Preserves client IP        • Rate limiting                 │
│                                                                 │
│   Use when:                    Use when:                       │
│   • Need raw performance       • Need smart routing            │
│   • Non-HTTP protocols         • Path-based routing            │
│   • Gaming, IoT, streaming     • Microservices                 │
│                                                                 │
│   Algorithms:                                                   │
│   ┌─────────────┬─────────────┬─────────────────────────────┐  │
│   │ Round Robin │ Least Conn  │ IP Hash (Sticky Sessions)   │  │
│   │     ↓       │     ↓       │            ↓                │  │
│   │  1→2→3→1    │  Route to   │  Same client → Same server  │  │
│   │             │  least busy │                             │  │
│   └─────────────┴─────────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### API Gateway Pattern

An API Gateway serves as the single entry point for all client requests to your backend services. Instead of clients knowing about and connecting to multiple services directly, they connect to one gateway that handles routing, security, and cross-cutting concerns.

**Why use an API Gateway?**

Without a gateway, every microservice must implement authentication, rate limiting, logging, and CORS handling independently. This leads to code duplication, inconsistent security, and complex client code that must track multiple service endpoints.

**Key responsibilities:**

- **Authentication/Authorization**: Verify JWT tokens or API keys once at the gateway rather than in every service
- **Rate Limiting**: Protect services from abuse by limiting requests per client
- **Request Routing**: Direct `/users/*` to the User Service, `/orders/*` to the Order Service
- **Protocol Translation**: Accept REST from clients, communicate via gRPC to internal services
- **Request/Response Transformation**: Add headers, modify payloads, aggregate responses from multiple services
- **Logging and Monitoring**: Central place to capture metrics and traces for all API traffic

**Trade-offs**: The gateway can become a bottleneck and single point of failure if not properly scaled. It also adds latency to every request (typically 1-10ms). Design for horizontal scaling and deploy across multiple availability zones.

```
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY                                │
│                                                                 │
│   ┌────────┐  ┌────────┐  ┌────────┐                           │
│   │ Mobile │  │  Web   │  │  IoT   │                           │
│   │  App   │  │ Client │  │ Device │                           │
│   └───┬────┘  └───┬────┘  └───┬────┘                           │
│       │           │           │                                 │
│       └───────────┼───────────┘                                 │
│                   │                                             │
│           ┌───────┴───────┐                                     │
│           │  API GATEWAY  │                                     │
│           │               │                                     │
│           │ • Auth/AuthZ  │  ← Verify tokens once              │
│           │ • Rate Limit  │  ← Protect services                │
│           │ • SSL Term    │  ← Offload encryption              │
│           │ • Routing     │  ← Direct to services              │
│           │ • Logging     │  ← Central observability           │
│           └───────┬───────┘                                     │
│                   │                                             │
│       ┌───────────┼───────────┐                                 │
│       │           │           │                                 │
│       ▼           ▼           ▼                                 │
│   ┌───────┐   ┌───────┐   ┌───────┐                            │
│   │ User  │   │ Order │   │Product│                            │
│   │Service│   │Service│   │Service│                            │
│   └───────┘   └───────┘   └───────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

### Rate Limiting Algorithms

Rate limiting protects your services from being overwhelmed by too many requests—whether from a misbehaving client, a DDoS attack, or simply unexpected viral traffic. Different algorithms offer different trade-offs between burst handling, fairness, and implementation complexity.

**Token Bucket** is the most common algorithm, allowing controlled bursts. Imagine a bucket that holds tokens, with new tokens added at a fixed rate. Each request consumes a token. If the bucket is empty, the request is rejected (or queued). The bucket capacity determines maximum burst size, while the refill rate determines sustained throughput.

*Example*: A bucket with 100 tokens refilled at 10 tokens/second allows bursts of 100 requests but sustains only 10 RPS.

**Leaky Bucket** enforces a constant output rate regardless of input burstiness. Requests enter the bucket, but they "leak" out at a fixed rate. If the bucket overflows, excess requests are discarded. This is ideal when your downstream services need predictable, smooth traffic.

**Fixed Window Counter** divides time into fixed windows (e.g., 1-minute intervals) and counts requests per window. Simple to implement but has a boundary problem: a user could send 100 requests at 0:59 and another 100 at 1:00, effectively getting 200 requests in 2 seconds while respecting the 100/minute limit.

**Sliding Window** solves the boundary problem by looking at a rolling time window. It can be implemented as a log of timestamps (accurate but memory-heavy) or as a weighted average of current and previous windows (efficient approximation).

```
┌─────────────────────────────────────────────────────────────────┐
│                   RATE LIMITING ALGORITHMS                      │
│                                                                 │
│  TOKEN BUCKET                    LEAKY BUCKET                   │
│  ┌─────────────────┐            ┌─────────────────┐            │
│  │ Tokens added at │            │ Requests enter  │            │
│  │   fixed rate    │            │   at any rate   │            │
│  │       ↓         │            │       ↓         │            │
│  │   ┌───────┐     │            │   ┌───────┐     │            │
│  │   │ ● ● ● │     │            │   │ ● ● ● │     │            │
│  │   │ ● ●   │ cap │            │   │ ● ●   │     │            │
│  │   │ ●     │     │            │   │ ●     │     │            │
│  │   └───┬───┘     │            │   └───┬───┘     │            │
│  │       │         │            │       │leak     │            │
│  │       ▼         │            │       ▼         │            │
│  │   Requests      │            │   Constant      │            │
│  │   (burst OK)    │            │   output rate   │            │
│  └─────────────────┘            └─────────────────┘            │
│  Allows bursts up to            Smooths traffic to             │
│  bucket capacity                constant rate                  │
│                                                                 │
│  FIXED WINDOW                    SLIDING WINDOW                 │
│  ┌─────────────────┐            ┌─────────────────┐            │
│  │                 │            │                 │            │
│  │  |████|████|    │            │    ████████     │            │
│  │  |    |    |    │            │   ←──────────→  │            │
│  │  t0   t1   t2   │            │  Rolling window │            │
│  │                 │            │                 │            │
│  │ Reset at window │            │ Smoother, no    │            │
│  │ boundary        │            │ boundary issue  │            │
│  │ (boundary spike │            │                 │            │
│  │  problem!)      │            │                 │            │
│  └─────────────────┘            └─────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Databases

### SQL vs NoSQL

The choice between SQL and NoSQL isn't about which is "better"—it's about which fits your use case. Understanding the fundamental differences helps you make the right choice.

**SQL databases** (MySQL, PostgreSQL, Oracle) store data in tables with predefined schemas. Every row has the same columns. They excel at complex queries with JOINs across multiple tables and provide strong ACID guarantees. However, they're traditionally harder to scale horizontally because JOINs across partitions are expensive.

*Choose SQL when*: You need transactions (banking, e-commerce), complex queries, or your data is highly relational with many-to-many relationships.

**NoSQL databases** embrace flexibility. They allow different "documents" to have different structures, making them ideal for rapidly evolving schemas. They're designed for horizontal scaling—add more nodes to handle more data or traffic. The trade-off is usually eventual consistency and limited query capabilities.

*Choose NoSQL when*: You need massive scale, flexible schemas, or your data access patterns are simple (key-based lookups rather than complex queries).

**Important nuance**: The lines are blurring. PostgreSQL now has excellent JSON support. Google Spanner offers SQL semantics with global scale. AWS Aurora provides MySQL compatibility with impressive scalability. Don't choose NoSQL just because "it scales"—modern SQL databases scale well for most use cases.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SQL vs NoSQL                                 │
│                                                                 │
│   SQL (Relational)               NoSQL (Non-relational)         │
│   ┌─────────────────┐            ┌─────────────────┐           │
│   │ ┌───┬───┬───┐   │            │   Key-Value     │           │
│   │ │ID │Name│Age│   │            │ ┌─────┬───────┐ │           │
│   │ ├───┼───┼───┤   │            │ │key1 │value1 │ │           │
│   │ │1  │John│25 │   │            │ │key2 │value2 │ │           │
│   │ │2  │Jane│30 │   │            │ └─────┴───────┘ │           │
│   │ └───┴───┴───┘   │            └─────────────────┘           │
│   └─────────────────┘                                          │
│                                                                 │
│   • Fixed schema                 • Flexible schema              │
│   • ACID compliant               • Eventual consistency (BASE) │
│   • Complex joins                • Simple queries               │
│   • Vertical scaling             • Horizontal scaling           │
│   • MySQL, PostgreSQL            • Redis, MongoDB, Cassandra    │
│                                                                 │
│   Choose SQL when:               Choose NoSQL when:             │
│   • Need transactions            • Need massive scale           │
│   • Complex queries/JOINs        • Flexible/evolving schema     │
│   • Data integrity critical      • Simple access patterns       │
│   • Relational data              • High write throughput        │
└─────────────────────────────────────────────────────────────────┘
```

### NoSQL Types

NoSQL isn't a single technology—it's a family of databases optimized for different data models and access patterns.

**Key-Value Stores** (Redis, DynamoDB) are the simplest: store a value with a key, retrieve it by key. Extremely fast for simple lookups. Redis extends this with data structures (lists, sets, sorted sets) making it great for caching, sessions, leaderboards, and real-time analytics.

**Document Databases** (MongoDB, Firestore) store semi-structured documents, typically JSON. Unlike relational tables, each document can have different fields. Ideal for content management, user profiles, or any domain where entities have varying attributes. They support querying on nested fields, not just keys.

**Graph Databases** (Neo4j, Amazon Neptune) model data as nodes and edges, optimized for traversing relationships. When you need to answer questions like "find friends of friends who also like jazz," graph databases outperform relational JOINs by orders of magnitude. Use for social networks, recommendation engines, fraud detection, and knowledge graphs.

**Columnar/Wide-Column Stores** (Cassandra, HBase, BigTable) organize data by columns rather than rows. When your query only needs 3 columns from a table with 100 columns, columnar storage reads only those 3 columns. This makes them excellent for analytics and aggregations over large datasets. They also excel at time-series data where you frequently query recent data.

```
┌─────────────────────────────────────────────────────────────────┐
│                      NoSQL TYPES                                │
│                                                                 │
│  KEY-VALUE              DOCUMENT              GRAPH             │
│  ┌───────────┐         ┌───────────┐         ┌───────────┐     │
│  │key → value│         │{          │         │  (A)──(B) │     │
│  │key → value│         │  "name":  │         │   │ \  │  │     │
│  │key → value│         │  "age":   │         │  (C)──(D) │     │
│  └───────────┘         │  "tags":[]│         └───────────┘     │
│                        │}          │                            │
│  Redis, DynamoDB       └───────────┘         Neo4j              │
│                        MongoDB               Amazon Neptune     │
│  Use: Cache,           Firestore                                │
│  sessions,             Use: CMS,             Use: Social nets,  │
│  leaderboards          catalogs,             recommendations,   │
│                        user profiles         fraud detection    │
│                                                                 │
│  COLUMNAR (Wide-Column)                                         │
│  ┌────────────────────────────────────────┐                    │
│  │ Row Key │ Col1  │ Col2  │ Col3  │ ... │                    │
│  │─────────┼───────┼───────┼───────┼─────│                    │
│  │ user1   │ name  │ email │       │     │  Sparse: rows can  │
│  │ user2   │ name  │       │ phone │     │  have diff columns │
│  └────────────────────────────────────────┘                    │
│  Cassandra, HBase, BigTable                                     │
│  Use: Analytics, time-series, IoT data, audit logs             │
└─────────────────────────────────────────────────────────────────┘
```

### Database Replication

Replication copies data across multiple servers for availability, durability, and read scalability. If one server fails, others continue serving requests. The key decision is how to handle writes.

**Master-Replica (Primary-Secondary)** is the most common pattern. One server (master) handles all writes and replicates changes to multiple replicas that handle reads. This scales read capacity linearly—add more replicas for more read throughput. The trade-off is replication lag: replicas may serve slightly stale data.

*Replication methods*:
- **Synchronous**: Master waits for replica acknowledgment before confirming write. Guarantees no data loss but adds latency.
- **Asynchronous**: Master confirms immediately, replicates in background. Faster writes but potential data loss if master fails before replication.

**Multi-AZ Deployment** protects against entire data center failures. Your primary database runs in Availability Zone A with a synchronous standby in Zone B. All writes go to primary; the standby continuously replays changes. If the primary fails, AWS automatically promotes the standby—typically within 60-120 seconds. Your application connects via a DNS endpoint that automatically points to the current primary.

*This is different from read replicas*: Multi-AZ is for high availability (automatic failover), while read replicas are for read scaling (manual promotion if needed).

```
┌─────────────────────────────────────────────────────────────────┐
│                 DATABASE REPLICATION                            │
│                                                                 │
│   MASTER-REPLICA (Read Replicas)                                │
│                                                                 │
│              Writes                                             │
│                 │                                               │
│                 ▼                                               │
│           ┌─────────┐                                           │
│           │ MASTER  │  ← All writes go here                    │
│           │  (RW)   │                                           │
│           └────┬────┘                                           │
│                │ Replication (sync or async)                    │
│       ┌────────┼────────┐                                       │
│       │        │        │                                       │
│       ▼        ▼        ▼                                       │
│   ┌───────┐┌───────┐┌───────┐                                  │
│   │Replica││Replica││Replica│  ← Handles reads                 │
│   │ (R)   ││ (R)   ││ (R)   │                                  │
│   └───────┘└───────┘└───────┘                                  │
│       ▲        ▲        ▲                                       │
│       └────────┼────────┘                                       │
│                │                                                │
│              Reads                                              │
│                                                                 │
│   MULTI-AZ DEPLOYMENT (High Availability)                       │
│   ┌─────────────────┐     ┌─────────────────┐                  │
│   │       AZ-A      │     │       AZ-B      │                  │
│   │   ┌─────────┐   │     │   ┌─────────┐   │                  │
│   │   │ PRIMARY │   │────►│   │ STANDBY │   │                  │
│   │   │   (RW)  │   │sync │   │  (Hot)  │   │                  │
│   │   └─────────┘   │     │   └─────────┘   │                  │
│   └─────────────────┘     └─────────────────┘                  │
│                                                                 │
│   Auto-failover: If primary fails, standby promoted            │
│   automatically in 60-120 seconds                              │
└─────────────────────────────────────────────────────────────────┘
```

### Database Sharding

When a single database server can't handle your data volume or write throughput, sharding (horizontal partitioning) distributes data across multiple database servers. Each shard holds a subset of the data.

**How sharding works**: A shard key (like user_id) determines which shard stores each record. When your application needs to read or write data, it calculates the shard from the key and connects to the appropriate database.

**Sharding strategies**:

- **Hash-based**: Apply a hash function to the key, modulo the number of shards. Distributes data evenly but makes range queries difficult (querying users with IDs 1-1000 might hit all shards).
  
- **Range-based**: Assign ranges to shards (A-H on shard 1, I-P on shard 2). Enables efficient range queries but can create hotspots if data isn't uniformly distributed.
  
- **Directory-based**: A lookup service maps keys to shards. Most flexible but adds a potential bottleneck and single point of failure.

**The challenge**: Cross-shard queries are expensive. If you need to JOIN data across shards, you're doing multiple queries and combining results in your application. Design your shard key to keep related data together. For example, in a multi-tenant SaaS app, shard by tenant_id so all queries within a tenant hit one shard.

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATABASE SHARDING                            │
│                                                                 │
│   Application makes request for user_id = 12345                 │
│                    │                                            │
│                    ▼                                            │
│           ┌─────────────────┐                                   │
│           │  Shard Router   │  ← Determines target shard       │
│           │  hash(12345) %3 │                                   │
│           │     = 0         │                                   │
│           └────────┬────────┘                                   │
│                    │                                            │
│       ┌────────────┼────────────┐                               │
│       │            │            │                               │
│       ▼            ▼            ▼                               │
│   ┌────────┐  ┌────────┐  ┌────────┐                           │
│   │Shard 0 │  │Shard 1 │  │Shard 2 │                           │
│   │────────│  │────────│  │────────│                           │
│   │hash=0  │  │hash=1  │  │hash=2  │                           │
│   │Users:  │  │Users:  │  │Users:  │                           │
│   │3,6,9...│  │1,4,7...│  │2,5,8...│                           │
│   └────────┘  └────────┘  └────────┘                           │
│                                                                 │
│   Sharding Strategies:                                          │
│   • Hash-based: Even distribution, poor range queries          │
│   • Range-based: Good ranges, risk of hotspots                 │
│   • Directory-based: Flexible, extra lookup step               │
│                                                                 │
│   Challenge: Cross-shard queries need aggregation in app       │
│   Solution: Choose shard key to keep related data together     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Caching

### Cache Architecture

A cache is a high-speed data storage layer that stores a subset of data so future requests can be served faster than querying the primary data store. Caches are typically stored in memory (RAM), which is orders of magnitude faster than disk-based databases.

**Why cache?** Consider an e-commerce product page. Without caching, every page view queries the database for product details, reviews, and recommendations. At scale, this overwhelms the database. With caching, the first request populates the cache; subsequent requests are served from memory in microseconds rather than milliseconds.

**Cache-aside (Lazy Loading)** is the most common pattern:
1. Application checks cache for data
2. **Cache hit**: Return cached data immediately
3. **Cache miss**: Query database, store result in cache, return to client

The application manages both the cache and database, giving full control over what gets cached and when. The downside is that the first request for any data always hits the database, and you must handle cache invalidation when data changes.

**Measuring cache effectiveness**: The cache hit ratio (percentage of requests served from cache) directly impacts performance. Calculate your Effective Access Time (EAT):

`EAT = (hit_ratio × cache_time) + (miss_ratio × db_time)`

With a 95% hit rate, 1ms cache time, and 100ms database time:
`EAT = (0.95 × 1) + (0.05 × 100) = 5.95ms`

Without caching: 100ms. With caching: ~6ms. That's a 17x improvement!

```
┌─────────────────────────────────────────────────────────────────┐
│                    CACHING ARCHITECTURE                         │
│                                                                 │
│   ┌──────┐        ┌───────────────┐        ┌──────────┐        │
│   │Client│───────►│  Application  │───────►│ Database │        │
│   └──────┘        │    Server     │        └──────────┘        │
│                   └───────┬───────┘              ▲              │
│                           │                      │              │
│                           ▼                      │              │
│                   ┌───────────────┐              │              │
│                   │     CACHE     │──────────────┘              │
│                   │  (Redis/MC)   │     Cache Miss              │
│                   └───────────────┘                             │
│                                                                 │
│   CACHE-ASIDE PATTERN (Lazy Loading)                            │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  1. App checks cache for key                             │  │
│   │  2. If HIT  → return cached data (fast path)            │  │
│   │  3. If MISS → query DB → write to cache → return        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Cache Performance Formula:                                    │
│   EAT = (Hit% × HitTime) + (Miss% × MissTime)                  │
│                                                                 │
│   Example: 95% hit rate, 1ms hit, 100ms miss                   │
│   EAT = (0.95 × 1) + (0.05 × 100) = 5.95ms                    │
│   Without cache: 100ms → With cache: ~6ms (17x faster!)        │
└─────────────────────────────────────────────────────────────────┘
```

### Caching Strategies

Different write strategies offer trade-offs between consistency, performance, and complexity.

**Write-Through** writes data to both cache and database synchronously. The write isn't considered complete until both succeed. This ensures cache and database are always consistent but adds latency to write operations since you're waiting for two writes.

*Use when*: Consistency is critical and write frequency is low (user profiles, configuration).

**Write-Back (Write-Behind)** writes to cache immediately and returns to the client. The cache asynchronously persists to the database later (often batched for efficiency). This provides excellent write performance but risks data loss—if the cache fails before persisting, updates are lost.

*Use when*: Write performance is critical and some data loss is acceptable (gaming leaderboards, analytics counters).

**Write-Around** bypasses the cache on writes, writing directly to the database. The cache is only populated on reads (cache-aside). This prevents the cache from being filled with data that might never be read, but the first read after a write always hits the database.

*Use when*: Data is written frequently but read rarely (logs, audit trails).

**Cache Invalidation** is the hardest problem in caching. When database data changes, the cache must be updated or invalidated. Strategies include:
- **TTL (Time-To-Live)**: Data expires after a set time. Simple but may serve stale data.
- **Active Invalidation**: Application explicitly deletes/updates cache when data changes.
- **Event-Driven**: Database changes trigger cache invalidation via messaging.

```
┌─────────────────────────────────────────────────────────────────┐
│                   CACHING STRATEGIES                            │
│                                                                 │
│  WRITE-THROUGH               WRITE-BACK (LAZY)                  │
│  ┌─────────────────┐        ┌─────────────────┐                │
│  │   Write to DB   │        │  Write to Cache │                │
│  │       AND       │        │      ONLY       │                │
│  │  Cache together │        │  (Async to DB)  │                │
│  │                 │        │                 │                │
│  │  App → Cache    │        │  App → Cache    │                │
│  │   ↓      ↓      │        │         ↓       │                │
│  │  DB ←────┘      │        │        DB       │                │
│  │  (synchronous)  │        │    (batched)    │                │
│  └─────────────────┘        └─────────────────┘                │
│  ✓ Always consistent        ✓ Fast writes                      │
│  ✗ Higher write latency     ✗ Risk of data loss               │
│                                                                 │
│  WRITE-AROUND                                                   │
│  ┌─────────────────┐                                           │
│  │  Write to DB    │                                           │
│  │  BYPASS Cache   │                                           │
│  │                 │                                           │
│  │  App ──────► DB │                                           │
│  │       Cache     │  Cache only populated on reads            │
│  │   (not updated) │                                           │
│  └─────────────────┘                                           │
│  ✓ No cache pollution from write-heavy, read-rare data        │
│  ✗ First read always misses                                   │
│                                                                 │
│  CACHE INVALIDATION (the hard problem):                        │
│  • TTL: Simple but may serve stale data                        │
│  • Active: App deletes cache on DB update                      │
│  • Events: DB changes → message → cache invalidation           │
└─────────────────────────────────────────────────────────────────┘
```

### Redis vs Memcached

Both Redis and Memcached are in-memory data stores used for caching, but they serve different use cases.

**Memcached** is the simpler, more focused solution. It stores string key-value pairs in memory, optimized for simplicity and raw performance. It's multi-threaded, making excellent use of multi-core systems. However, it lacks persistence (data is lost on restart), replication, and advanced data structures.

*Choose Memcached when*: You need simple caching of large objects, you're already multi-threaded, and you don't need persistence or complex operations.

**Redis** is a full-featured data structure server. Beyond simple strings, it supports lists (for queues), sets (for unique collections), sorted sets (for leaderboards), hashes (for objects), and more. Redis provides persistence through RDB snapshots and AOF (append-only file), built-in replication, and Lua scripting for complex atomic operations.

*Choose Redis when*: You need data structures beyond simple strings, want persistence, need pub/sub messaging, or require atomic operations on complex data.

**Performance note**: Redis is single-threaded (by design, to avoid locking complexity), but it's so fast that this rarely matters—a single Redis instance can handle 100,000+ operations per second. For higher throughput, run multiple Redis instances or use Redis Cluster.

```
┌─────────────────────────────────────────────────────────────────┐
│                   REDIS vs MEMCACHED                            │
│                                                                 │
│   ┌─────────────────────────┬─────────────────────────┐        │
│   │         REDIS           │       MEMCACHED         │        │
│   ├─────────────────────────┼─────────────────────────┤        │
│   │ Data Types:             │ Data Types:             │        │
│   │ • Strings               │ • Strings only          │        │
│   │ • Lists (queues)        │                         │        │
│   │ • Sets (unique items)   │                         │        │
│   │ • Hashes (objects)      │                         │        │
│   │ • Sorted Sets (ranks)   │                         │        │
│   ├─────────────────────────┼─────────────────────────┤        │
│   │ Persistence: YES        │ Persistence: NO         │        │
│   │ • RDB Snapshots         │ (data lost on restart)  │        │
│   │ • AOF (append-only)     │                         │        │
│   ├─────────────────────────┼─────────────────────────┤        │
│   │ Replication: Built-in   │ Replication: No         │        │
│   ├─────────────────────────┼─────────────────────────┤        │
│   │ Threading: Single       │ Threading: Multi        │        │
│   │ (but still 100K+ ops/s) │ (better multicore use)  │        │
│   ├─────────────────────────┼─────────────────────────┤        │
│   │ Use: Sessions, queues,  │ Use: Simple caching,    │        │
│   │ leaderboards, pub/sub   │ large objects (>100KB)  │        │
│   └─────────────────────────┴─────────────────────────┘        │
│                                                                 │
│   Rule of thumb: Start with Redis unless you have a specific   │
│   reason to use Memcached (simpler ops, multi-threaded needs)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Message Queues & Pub/Sub

### Message Queue Architecture

Message queues decouple components by allowing asynchronous communication. Instead of Service A calling Service B directly (synchronous), Service A puts a message in a queue and continues its work. Service B processes the message when ready.

**Why use queues?**
- **Decoupling**: Services don't need to know about each other
- **Buffering**: Handle traffic spikes by queuing excess requests
- **Reliability**: Messages persist until processed, surviving service restarts
- **Scalability**: Add more consumers to process messages faster

**Queue vs Pub/Sub** are fundamentally different patterns:

In a **Queue**, each message is delivered to exactly one consumer. When multiple consumers listen to the same queue, they compete for messages—each message goes to only one of them. This is perfect for task distribution (process these 1000 images).

In **Pub/Sub**, each message is delivered to all subscribers of a topic. When a user posts a tweet, multiple services need to know: the notification service, the analytics service, the timeline service. Each gets its own copy of the message.

**Dead Letter Queue (DLQ)** handles messages that repeatedly fail processing. After N retries, the message moves to the DLQ for manual inspection rather than blocking the queue or being lost. Common causes: malformed data, bugs in consumer code, or downstream service failures.

```
┌─────────────────────────────────────────────────────────────────┐
│                   MESSAGE QUEUE                                 │
│                                                                 │
│   ┌──────────┐      ┌─────────────────┐      ┌──────────┐      │
│   │ Producer │─────►│     QUEUE       │─────►│ Consumer │      │
│   └──────────┘      │  ┌───┬───┬───┐  │      └──────────┘      │
│                     │  │ 1 │ 2 │ 3 │  │                        │
│                     │  └───┴───┴───┘  │                        │
│                     └─────────────────┘                        │
│                                                                 │
│   Point-to-Point: Each message goes to ONE consumer            │
│   Use: Task distribution, work queues, job processing          │
│                                                                 │
│                   PUB/SUB (Publish-Subscribe)                   │
│                                                                 │
│   ┌──────────┐      ┌─────────────────┐      ┌──────────┐      │
│   │Publisher │─────►│     TOPIC       │─────►│Subscriber│      │
│   └──────────┘      │    (Broker)     │──┐   └──────────┘      │
│                     └─────────────────┘  │                      │
│                                          │   ┌──────────┐      │
│                                          └──►│Subscriber│      │
│                                              └──────────┘      │
│                                                                 │
│   Broadcast: Each message goes to ALL subscribers              │
│   Use: Event notification, real-time updates, fan-out          │
└─────────────────────────────────────────────────────────────────┘
```

### Kafka Architecture

Apache Kafka is a distributed streaming platform designed for high-throughput, fault-tolerant message handling. Unlike traditional queues, Kafka persists messages to disk and allows consumers to "replay" historical messages.

**Core concepts**:

**Topics** are categories for messages (think: database tables). A topic is split into **partitions** distributed across brokers. Each partition is an ordered, immutable sequence of messages.

**Partitions** enable parallelism. If a topic has 10 partitions, you can have up to 10 consumers reading in parallel. Messages within a partition maintain order, but there's no global ordering across partitions.

**Consumer Groups** enable both pub/sub and queue semantics. Consumers in the same group share partitions (queue behavior—each message to one consumer). Different consumer groups each receive all messages (pub/sub behavior).

**Offsets** track consumer progress. Each message has an offset (position in partition). Consumers commit offsets to mark progress. If a consumer restarts, it resumes from the last committed offset—no messages lost.

**Why Kafka over RabbitMQ?**
- Kafka: High throughput, message replay, stream processing. Better for analytics, logs, event sourcing.
- RabbitMQ: Lower latency, complex routing, traditional messaging. Better for task queues, RPC patterns.

```
┌─────────────────────────────────────────────────────────────────┐
│                    KAFKA ARCHITECTURE                           │
│                                                                 │
│   ┌──────────┐                              ┌──────────┐       │
│   │Producer 1│──┐                       ┌──►│Consumer 1│       │
│   └──────────┘  │                       │   └──────────┘       │
│   ┌──────────┐  │    ┌─────────────┐    │   ┌──────────┐       │
│   │Producer 2│──┼───►│   TOPIC     │────┼──►│Consumer 2│       │
│   └──────────┘  │    │             │    │   └──────────┘       │
│   ┌──────────┐  │    │ Partition 0 │    │   ┌──────────┐       │
│   │Producer 3│──┘    │ ┌─┬─┬─┬─┬─┐│    └──►│Consumer 3│       │
│   └──────────┘       │ │0│1│2│3│4││        └──────────┘       │
│                      │ └─┴─┴─┴─┴─┘│                            │
│                      │  ↑ offset   │       Consumer Group       │
│                      │             │       (share partitions)   │
│                      │ Partition 1 │                            │
│                      │ ┌─┬─┬─┬─┐  │                            │
│                      │ │0│1│2│3│  │  Messages ordered WITHIN   │
│                      │ └─┴─┴─┴─┘  │  partition, not across     │
│                      │             │                            │
│                      │ Partition 2 │                            │
│                      │ ┌─┬─┬─┐    │  Messages persisted to     │
│                      │ │0│1│2│    │  disk, replayable          │
│                      │ └─┴─┴─┘    │                            │
│                      └─────────────┘                            │
│                                                                 │
│   Key features:                                                 │
│   • High throughput (millions of messages/sec)                 │
│   • Durable (persisted to disk, configurable retention)        │
│   • Replayable (consumers can re-read historical messages)     │
│   • Scalable (add partitions and brokers)                      │
└─────────────────────────────────────────────────────────────────┘
```

### Dead Letter Queue

When message processing fails repeatedly, you need a strategy to prevent one "poison pill" message from blocking your entire queue. Dead Letter Queues (DLQs) solve this by moving problematic messages aside for investigation.

**How it works**: Configure a retry policy (e.g., 3 attempts with exponential backoff). If a message fails all retries, the queue automatically moves it to the DLQ. Your main queue continues processing other messages while engineers investigate the DLQ contents.

**What causes DLQ messages?**
- **Malformed data**: A message with unexpected format that causes parsing errors
- **Business logic failures**: Data that violates constraints (e.g., negative quantities)
- **Downstream failures**: A dependent service is down (though this might warrant infinite retries)
- **Bugs**: Consumer code has a bug handling certain edge cases

**Best practices**:
- Monitor DLQ depth—it should be empty or near-empty
- Set up alerts when messages enter the DLQ
- Include debugging metadata (original timestamp, failure reason, retry count)
- Build tooling to replay DLQ messages after fixing issues

```
┌─────────────────────────────────────────────────────────────────┐
│                   DEAD LETTER QUEUE (DLQ)                       │
│                                                                 │
│   ┌──────────┐      ┌─────────────┐      ┌──────────┐          │
│   │ Producer │─────►│ Main Queue  │─────►│ Consumer │          │
│   └──────────┘      └──────┬──────┘      └────┬─────┘          │
│                            │                   │                │
│                            │    Retry 1:  ✗   │                │
│                            │    Retry 2:  ✗   │                │
│                            │    Retry 3:  ✗   │                │
│                            │                   │                │
│                            ▼                   │                │
│                     ┌──────────────┐           │                │
│                     │  Dead Letter │◄──────────┘                │
│                     │    Queue     │  After max retries         │
│                     └──────┬───────┘                            │
│                            │                                    │
│                            ▼                                    │
│                    Manual inspection                            │
│                    Fix issue → Replay                           │
│                                                                 │
│   DLQ messages caused by:                                       │
│   • Malformed data (parsing errors)                            │
│   • Business rule violations                                   │
│   • Consumer bugs on edge cases                                │
│   • Downstream service failures (maybe infinite retry instead) │
│                                                                 │
│   Best practices:                                               │
│   • Monitor DLQ depth (should be ~0)                           │
│   • Alert on new DLQ messages                                  │
│   • Include debugging metadata                                 │
│   • Build replay tooling                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Storage

### Storage Types Comparison

Cloud storage comes in three fundamental types, each optimized for different access patterns and use cases.

**Block Storage (EBS)** presents storage as raw blocks, like a hard drive attached to your computer. The operating system formats it with a file system (ext4, NTFS) and mounts it as a volume. Block storage offers the lowest latency and highest IOPS, making it ideal for databases and operating systems. However, it can only be attached to one EC2 instance at a time (within the same AZ).

*Use for*: Boot volumes, databases, applications requiring raw device access.

**File Storage (EFS/FSx)** provides a managed file system accessible via standard protocols (NFS for Linux, SMB for Windows). Multiple instances can mount the same file system simultaneously, seeing the same files. It scales automatically and replicates across AZs.

*Use for*: Shared application data, content management, big data analytics, container storage.

**Object Storage (S3)** stores data as objects in a flat namespace (buckets). Each object consists of data, metadata, and a unique key. Objects are accessed via HTTP APIs, not mounted as file systems. S3 offers virtually unlimited capacity and exceptional durability (99.999999999%—eleven 9s).

*Use for*: Static assets (images, videos), backups, data lakes, static website hosting. NOT for operating systems or databases (no POSIX interface, higher latency).

```
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE TYPES                                │
│                                                                 │
│   BLOCK STORAGE (EBS)                                           │
│   ┌─────────────────────────────────────┐                      │
│   │ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐      │  Raw blocks like a   │
│   │ │ 1 │ │ 2 │ │ 3 │ │ 4 │ │ 5 │ ...  │  hard drive          │
│   │ └───┘ └───┘ └───┘ └───┘ └───┘      │                      │
│   └─────────────────────────────────────┘                      │
│   • Attached to single EC2 (same AZ)    • Lowest latency       │
│   • Format with file system             • Best for DBs, OS     │
│                                                                 │
│   FILE STORAGE (EFS/FSx)                                        │
│   ┌─────────────────────────────────────┐                      │
│   │         /root                        │  Hierarchical file  │
│   │        /     \                       │  system (NFS/SMB)   │
│   │      /dir1   /dir2                   │                      │
│   │      /   \      \                    │                      │
│   │   file1 file2  file3                 │                      │
│   └─────────────────────────────────────┘                      │
│   • Shared across instances             • Regional scope       │
│   • Auto-scales                         • Good for CMS, data   │
│                                                                 │
│   OBJECT STORAGE (S3)                                           │
│   ┌─────────────────────────────────────┐                      │
│   │  BUCKET                              │  Flat key-value     │
│   │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐   │  accessed via API    │
│   │  │obj1 │ │obj2 │ │obj3 │ │obj4 │   │                      │
│   │  └─────┘ └─────┘ └─────┘ └─────┘   │                      │
│   └─────────────────────────────────────┘                      │
│   • Unlimited capacity                  • 11 9s durability     │
│   • HTTP API access                     • NOT for OS or DBs    │
│   • Use for: media, backups, data lakes                        │
└─────────────────────────────────────────────────────────────────┘
```

### Storage Performance Comparison

Understanding storage performance helps you choose the right type for your workload.

**IOPS (I/O Operations Per Second)** measures how many read/write operations storage can handle. Databases with many small transactions need high IOPS.

**Throughput** measures data transfer rate (MB/s). Large file transfers or streaming need high throughput.

**Latency** measures the delay between requesting data and receiving it. Real-time applications need low latency.

Block storage (EBS) provides the lowest latency (sub-millisecond for SSD-backed volumes) and can be provisioned for high IOPS (up to 256,000 for io2 volumes). It's limited to one EC2 instance but delivers consistent, predictable performance.

File storage (EFS) adds a network hop, increasing latency slightly. However, it supports concurrent access from thousands of instances, and throughput scales with file system size.

Object storage (S3) has the highest latency (typically 100-200ms) because it's accessed over HTTP. However, it offers massive aggregate throughput—you can read many objects in parallel from many clients simultaneously.

**Cost-performance trade-off**: S3 is cheapest, followed by EFS, then EBS. Choose based on access patterns: S3 for infrequent access to large data, EBS for frequent access to hot data, EFS when you need shared access.

```
┌─────────────────────────────────────────────────────────────────┐
│                 STORAGE PERFORMANCE                             │
│                                                                 │
│   Latency      Block        File         Object                 │
│      ▲         (EBS)        (EFS)         (S3)                  │
│      │                                                          │
│   High│                                    ●  ~100-200ms        │
│      │                                                          │
│   Med │                       ●  ~1-10ms                        │
│      │                                                          │
│   Low │           ●  <1ms                                       │
│      │                                                          │
│      └──────────────────────────────────────►                   │
│                    Concurrent Access                            │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │ Metric      │  Block (EBS) │  File (EFS) │ Object (S3) │  │
│   ├─────────────┼──────────────┼─────────────┼─────────────┤  │
│   │ Latency     │  <1ms        │  1-10ms     │ 100-200ms   │  │
│   │ Max IOPS    │  256,000     │  Scales     │ N/A         │  │
│   │ Throughput  │  4,000 MB/s  │  Scales     │ Unlimited*  │  │
│   │ Access      │  Single EC2  │  Multiple   │ Via API     │  │
│   │ Scope       │  AZ          │  Regional   │ Regional    │  │
│   │ Cost        │  $$          │  $$$        │ $           │  │
│   └─────────────┴──────────────┴─────────────┴─────────────┘  │
│                                                                 │
│   * S3 throughput unlimited in aggregate (parallel requests)   │
│                                                                 │
│   Choose: EBS for DBs, EFS for shared files, S3 for archives  │
└─────────────────────────────────────────────────────────────────┘
```

### CDN Architecture

A Content Delivery Network (CDN) caches content at edge locations around the world, serving users from the nearest location rather than your origin server. This dramatically reduces latency and offloads traffic from your infrastructure.

**How it works**: When a user in Tokyo requests an image from your US-based server, without a CDN they'd wait for a round-trip across the Pacific (~150ms just for network latency). With a CDN, the request goes to a Tokyo edge server. If the content is cached, it's served immediately (~5ms). If not, the edge server fetches it from your origin, caches it, and serves it—subsequent Tokyo users get the cached version.

**Push vs Pull CDN**:
- **Pull (Origin Pull)**: The CDN fetches content from your origin when first requested, then caches it. Simple to set up—just point the CDN at your origin. Content may be stale for the TTL duration.
- **Push (Origin Push)**: You proactively upload content to the CDN before users request it. Better for large files or predictable content but requires more management.

**CDN benefits beyond caching**:
- **DDoS protection**: Edge servers absorb attack traffic, protecting your origin
- **SSL termination**: Handle HTTPS at the edge, reducing certificate management overhead
- **Image optimization**: Automatically resize and compress images per device
- **Geographic load balancing**: Route users to the nearest healthy origin

**AWS CloudFront** integrates with Shield (DDoS), WAF (web application firewall), and Lambda@Edge (run code at edge locations).

```
┌─────────────────────────────────────────────────────────────────┐
│                    CDN ARCHITECTURE                             │
│                                                                 │
│   Without CDN: User → 150ms → Origin → 150ms → User (300ms)    │
│   With CDN:    User → 5ms → Edge (cache hit!) → User (5ms)     │
│                                                                 │
│   ┌──────┐                                        ┌──────┐     │
│   │User A│ (Tokyo)                        (NYC)   │User B│     │
│   └──┬───┘                                        └──┬───┘     │
│      │                                               │          │
│      ▼                                               ▼          │
│   ┌─────────┐                                  ┌─────────┐     │
│   │  Edge   │  Cache hit?                      │  Edge   │     │
│   │ Server  │  Yes → Serve immediately         │ Server  │     │
│   │ (Tokyo) │  No  → Fetch from origin         │ (NYC)   │     │
│   └────┬────┘                                  └────┬────┘     │
│        │              Cache Miss                    │          │
│        │                  │                         │          │
│        └──────────────────┼─────────────────────────┘          │
│                           │                                     │
│                           ▼                                     │
│                    ┌────────────┐                               │
│                    │   ORIGIN   │                               │
│                    │  SERVER    │                               │
│                    │ (S3/EC2)   │                               │
│                    └────────────┘                               │
│                                                                 │
│   CDN Benefits:                                                 │
│   • Reduced latency (serve from nearby edge)                   │
│   • Offload origin (80%+ traffic from cache)                   │
│   • DDoS protection (absorb attacks at edge)                   │
│   • SSL termination at edge                                    │
│                                                                 │
│   Push CDN: Upload content proactively (predictable content)   │
│   Pull CDN: Fetch on first request (most common, simpler)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scalability Patterns

### Horizontal vs Vertical Scaling

When your system can't handle the load, you have two fundamental approaches: make machines bigger or add more machines.

**Vertical Scaling (Scale Up)** means adding more resources to existing servers—more CPU, more RAM, faster disks. It's the simpler approach: no code changes, no distributed systems complexity. Your application runs on one powerful machine.

*Limitations*: Hardware has physical limits (you can't buy a 1000-core CPU). It's expensive—high-end servers cost disproportionately more. And it's a single point of failure—if that one server dies, everything dies.

**Horizontal Scaling (Scale Out)** means adding more servers. Instead of one big machine, you run many smaller machines behind a load balancer. This approach has no theoretical limit—need more capacity? Add more servers.

*Requirements*: Your application must be designed for horizontal scaling. State must be externalized (sessions in Redis, not in memory). Requests must be stateless or use sticky sessions. Database scaling requires additional strategies (read replicas, sharding).

**The reality**: Most systems use both. Scale vertically until it's too expensive or risky, then scale horizontally. A common pattern is vertical scaling for databases (easier consistency) and horizontal scaling for stateless application servers.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCALING STRATEGIES                           │
│                                                                 │
│   VERTICAL (Scale Up)              HORIZONTAL (Scale Out)       │
│                                                                 │
│   ┌─────────────┐                 ┌───┐ ┌───┐ ┌───┐ ┌───┐      │
│   │             │                 │   │ │   │ │   │ │   │      │
│   │             │                 │ S │ │ S │ │ S │ │ S │      │
│   │   BIGGER    │                 │ E │ │ E │ │ E │ │ E │      │
│   │   SERVER    │                 │ R │ │ R │ │ R │ │ R │      │
│   │             │                 │ V │ │ V │ │ V │ │ V │      │
│   │  More CPU   │                 │ E │ │ E │ │ E │ │ E │      │
│   │  More RAM   │                 │ R │ │ R │ │ R │ │ R │      │
│   │             │                 │   │ │   │ │   │ │   │      │
│   └─────────────┘                 └───┘ └───┘ └───┘ └───┘      │
│                                                                 │
│   Pros:                           Pros:                         │
│   ✓ Simple, no code changes       ✓ No theoretical limit        │
│   ✓ No distributed complexity     ✓ Fault tolerant              │
│   ✓ Easier consistency            ✓ Cost-effective at scale     │
│                                                                 │
│   Cons:                           Cons:                         │
│   ✗ Hardware limits               ✗ Stateless design required   │
│   ✗ Expensive at high end         ✗ Distributed complexity      │
│   ✗ Single point of failure       ✗ Data consistency harder     │
│                                                                 │
│   Reality: Use both. Vertical for DBs, horizontal for apps.    │
└─────────────────────────────────────────────────────────────────┘
```

### Microservices Architecture

Microservices decompose a monolithic application into small, independent services that communicate over the network. Each service owns its data and can be developed, deployed, and scaled independently.

**Why microservices?**
- **Independent scaling**: Scale only the services that need it. If your image processing is bottleneck, scale just that service.
- **Technology diversity**: Each service can use the best tool for its job—Python for ML, Go for performance, Node for real-time.
- **Fault isolation**: A bug in one service doesn't crash the whole system.
- **Team autonomy**: Small teams own services end-to-end, moving faster without coordination overhead.

**The challenges are significant**:
- **Network complexity**: What was a function call becomes a network request that can fail, timeout, or return errors.
- **Data consistency**: Without a shared database, maintaining consistency across services requires careful design (sagas, eventual consistency).
- **Operational overhead**: More services mean more deployments, more logs, more monitoring, more things that can fail.
- **Debugging difficulty**: A user request might touch 10 services—tracing issues requires distributed tracing (Jaeger, Zipkin).

**When to use microservices**: Large teams working on a large application with different scaling needs. NOT for startups or small teams—start with a monolith and extract services when needed.

```
┌─────────────────────────────────────────────────────────────────┐
│                  MICROSERVICES ARCHITECTURE                     │
│                                                                 │
│   ┌────────────────────────────────────────────────────────┐   │
│   │                    API GATEWAY                          │   │
│   │         (Auth, Rate Limiting, Routing)                  │   │
│   └───────────────────────┬────────────────────────────────┘   │
│                           │                                     │
│       ┌───────────────────┼───────────────────┐                │
│       │                   │                   │                │
│       ▼                   ▼                   ▼                │
│   ┌───────┐          ┌───────┐          ┌───────┐             │
│   │ User  │          │ Order │          │Product│             │
│   │Service│          │Service│          │Service│             │
│   │ Team A│          │ Team B│          │ Team C│             │
│   └───┬───┘          └───┬───┘          └───┬───┘             │
│       │                  │                   │                 │
│       ▼                  ▼                   ▼                 │
│   ┌───────┐          ┌───────┐          ┌───────┐             │
│   │User DB│          │OrderDB│          │ProdDB │             │
│   └───────┘          └───────┘          └───────┘             │
│                                                                 │
│   Benefits:                     Challenges:                    │
│   ✓ Independent scaling         ✗ Network latency/failures    │
│   ✓ Technology flexibility      ✗ Data consistency            │
│   ✓ Fault isolation             ✗ Operational complexity      │
│   ✓ Team autonomy               ✗ Debugging (need tracing)    │
│                                                                 │
│   Start with a monolith. Extract services when there's clear  │
│   benefit—don't start with microservices!                      │
└─────────────────────────────────────────────────────────────────┘
```

### Disaster Recovery

Disaster Recovery (DR) plans for the worst: entire region failures, natural disasters, or catastrophic bugs. Your DR strategy depends on two key metrics:

**RTO (Recovery Time Objective)**: Maximum acceptable downtime. How long can your business survive without the system? E-commerce during Black Friday might need minutes; an internal reporting tool might tolerate hours.

**RPO (Recovery Point Objective)**: Maximum acceptable data loss. How much data can you afford to lose? A social media platform might accept losing the last hour of posts; a bank can't lose any transactions.

**DR Strategies** (from least to most expensive):

**Backup & Restore**: Regular backups to another region. Cheapest but slowest recovery (hours). RPO = time since last backup.

**Pilot Light**: Minimal core infrastructure running in DR region (databases replicating). On disaster, spin up remaining infrastructure. Recovery in tens of minutes.

**Warm Standby**: Scaled-down but functional environment in DR region. Traffic can be routed there immediately. Recovery in minutes.

**Hot Standby (Multi-Site)**: Full production capacity in both regions, traffic split between them. Instant failover, zero data loss. Most expensive but best RTO/RPO.

Route 53 (DNS) handles failover by directing traffic to the healthy region based on health checks.

```
┌─────────────────────────────────────────────────────────────────┐
│                  DISASTER RECOVERY                              │
│                                                                 │
│   RTO: Recovery Time Objective (max downtime you can tolerate) │
│   RPO: Recovery Point Objective (max data loss you can accept) │
│                                                                 │
│   PRIMARY REGION (us-east-1)     SECONDARY REGION (us-west-2)  │
│   ┌─────────────────────────┐    ┌─────────────────────────┐   │
│   │                         │    │                         │   │
│   │  ┌─────┐    ┌─────┐    │    │  ┌─────┐    ┌─────┐    │   │
│   │  │ EC2 │    │ EC2 │    │    │  │ EC2 │    │ EC2 │    │   │
│   │  └──┬──┘    └──┬──┘    │    │  └──┬──┘    └──┬──┘    │   │
│   │     │          │        │    │     │          │        │   │
│   │     └────┬─────┘        │    │     └────┬─────┘        │   │
│   │          │              │    │          │              │   │
│   │     ┌────┴────┐         │    │     ┌────┴────┐         │   │
│   │     │   RDS   │─────────┼────┼────►│   RDS   │         │   │
│   │     │ Primary │  Async  │    │     │ Standby │         │   │
│   │     └─────────┘  Repli- │    │     └─────────┘         │   │
│   │                  cation │    │                         │   │
│   └─────────────────────────┘    └─────────────────────────┘   │
│              │                              │                   │
│              └──────────────┬───────────────┘                   │
│                             │                                   │
│                      ┌──────┴──────┐                           │
│                      │  Route 53   │  DNS health check         │
│                      │ (Failover)  │  auto-switches traffic    │
│                      └─────────────┘                           │
│                                                                 │
│   Strategy      │ RTO      │ RPO      │ Cost                   │
│   ─────────────────────────────────────────────                │
│   Backup/Restore│ Hours    │ Hours    │ $                      │
│   Pilot Light   │ 10s mins │ Minutes  │ $$                     │
│   Warm Standby  │ Minutes  │ Minutes  │ $$$                    │
│   Hot Standby   │ Seconds  │ Zero     │ $$$$                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Distributed System Patterns

### Consistent Hashing

Standard hashing (hash(key) % N) has a major problem: when you add or remove a node, almost all keys need to move. Consistent hashing solves this by mapping both keys and nodes onto a ring.

**How it works**: Imagine a ring representing the full hash space (0 to 2^32-1). Nodes are placed on the ring at positions determined by hashing their identifiers. Keys are also hashed to positions. Each key belongs to the first node encountered when moving clockwise from the key's position.

**Why it's better**: When a node joins or leaves, only keys between it and its predecessor need to move—roughly 1/N of the keys instead of nearly all of them. This minimizes disruption during scaling or failures.

**Virtual nodes** improve distribution. Instead of each physical node having one position, give it many "virtual" positions. This ensures more even distribution of keys, especially with heterogeneous node capacities.

```
┌─────────────────────────────────────────────────────────────────┐
│                  CONSISTENT HASHING                             │
│                                                                 │
│   Standard hashing: hash(key) % N                               │
│   Problem: If N changes, almost ALL keys move!                 │
│                                                                 │
│   Consistent hashing: Both keys and nodes on a ring            │
│                                                                 │
│                        0                                        │
│                        │                                        │
│              ┌─────────┴─────────┐                              │
│             /                     \                             │
│            /      Node A           \                            │
│           │         ● (hash: 100)   │                           │
│           │    ★ key1 (hash: 80)    │  key1 → clockwise → A    │
│    270 ───┤                         ├─── 90                     │
│           │                         │                           │
│           │    ●                 ●  │                           │
│           │  Node C          Node B │                           │
│            \  (240)           (180) /                           │
│             \                     /                             │
│              └─────────┬─────────┘                              │
│                        │                                        │
│                       180                                       │
│                                                                 │
│   When Node B leaves: Only keys between C and B move to A      │
│   When Node D joins: Only keys in its range move to D          │
│                                                                 │
│   Virtual nodes: Each physical node → multiple ring positions  │
│   Benefits: More even distribution, handle capacity differences│
└─────────────────────────────────────────────────────────────────┘
```

### Quorum

Quorum-based systems balance consistency and availability by requiring a minimum number of nodes to participate in reads and writes. This ensures that reads and writes have at least one node in common.

**The formula**: For N replicas, configure W (write quorum) and R (read quorum). If W + R > N, every read will see at least one node that participated in the most recent write, ensuring strong consistency.

**Common configurations**:
- **W=N, R=1**: Strong consistency on writes, fast reads. But writes fail if any node is down.
- **W=1, R=N**: Fast writes, but reads are slow and must query all nodes.
- **W=majority, R=majority**: Balanced. For N=3, W=2, R=2 allows one node failure while maintaining consistency.

**Eventual consistency** (W=1, R=1): Fastest but no consistency guarantee. The write might go to node A while the read goes to node B, which hasn't replicated yet.

```
┌─────────────────────────────────────────────────────────────────┐
│                      QUORUM                                     │
│                                                                 │
│   N = Total replicas                                           │
│   W = Write quorum (nodes acknowledging write)                 │
│   R = Read quorum (nodes responding to read)                   │
│                                                                 │
│   RULE: W + R > N  →  Guarantees overlap (strong consistency)  │
│                                                                 │
│   Example: N=3, W=2, R=2  →  2+2=4 > 3 ✓                       │
│                                                                 │
│   WRITE (W=2 must ACK)          READ (R=2 must respond)        │
│   ┌───┐ ┌───┐ ┌───┐              ┌───┐ ┌───┐ ┌───┐            │
│   │ ✓ │ │ ✓ │ │   │              │ ✓ │ │ ✓ │ │   │            │
│   │N1 │ │N2 │ │N3 │              │N1 │ │N2 │ │N3 │            │
│   └───┘ └───┘ └───┘              └───┘ └───┘ └───┘            │
│     │     │                        │     │                     │
│     └──┬──┘                        └──┬──┘                     │
│        │                              │                        │
│    W=2 ✓                          R=2 ✓                        │
│                                                                 │
│   N1 and N2 overlap → Read sees latest write!                  │
│                                                                 │
│   Configurations:                                               │
│   • W=N, R=1: Strong write, one node down blocks writes       │
│   • W=1, R=N: Fast writes, must read all nodes                │
│   • W=majority, R=majority: Balanced (recommended)             │
│   • W=1, R=1: Eventual consistency (no guarantee)              │
└─────────────────────────────────────────────────────────────────┘
```

### Leader Election

Many distributed systems need a single "leader" to coordinate activities—processing writes, assigning work, or making decisions. Leader election algorithms ensure exactly one leader is chosen, even as nodes fail and recover.

**Why it's hard**: In a distributed system, nodes can't see each other's state directly. Network partitions might make it appear a leader is dead when it's actually fine. Having two leaders (split-brain) can cause data corruption.

**Raft** is a popular consensus algorithm that's easier to understand than Paxos. Nodes are either leaders, followers, or candidates. Leaders send heartbeats; if followers don't hear from the leader, they become candidates and request votes. The candidate with majority votes becomes the new leader.

**Coordination services** like ZooKeeper and etcd implement leader election so you don't have to. They provide strongly consistent key-value storage with features like ephemeral nodes (disappear when client disconnects) and watches (notify when data changes). Use these rather than implementing consensus yourself!

```
┌─────────────────────────────────────────────────────────────────┐
│                   LEADER ELECTION                               │
│                                                                 │
│   Why: Coordinate writes, assign work, make decisions          │
│   Challenge: Network partitions can cause "split brain"        │
│                                                                 │
│   ┌───────┐     ┌───────┐     ┌───────┐                        │
│   │Node A │     │Node B │     │Node C │                        │
│   │Follower│    │Follower│    │Follower│                       │
│   └───┬───┘     └───┬───┘     └───┬───┘                        │
│       │             │             │                             │
│       │   Leader timeout!         │                             │
│       │   A becomes candidate     │                             │
│       │             │             │                             │
│       │──Request vote────────────►│                             │
│       │◄─Vote yes─────────────────│                             │
│       │──Request vote───►│        │                             │
│       │◄─Vote yes────────│        │                             │
│       │             │             │                             │
│       │   Majority! A is leader   │                             │
│       │             │             │                             │
│       ▼             ▼             ▼                             │
│   ┌───────┐     ┌───────┐     ┌───────┐                        │
│   │ LEADER│     │Follower│    │Follower│                        │
│   └───────┘     └───────┘     └───────┘                        │
│       │             │             │                             │
│       │──Heartbeat──────────────►│  Leader sends heartbeats    │
│       │──Heartbeat───►│          │  Followers reset timeout    │
│                                                                 │
│   Don't implement yourself! Use:                               │
│   • ZooKeeper: Mature, battle-tested                           │
│   • etcd: Simpler, Kubernetes uses it                          │
│   • Consul: Also provides service discovery                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Capacity Estimation

### Key Latencies & QPS

**QPS** (queries per second) is the standard rate metric. Before designing a system, you need rough estimates for capacity planning. These reference numbers help you reason about bottlenecks.

**Why key-value stores are faster than SQL**: Key-value stores like Redis use simple `GET/PUT` operations with O(1) hash lookups. SQL databases must parse queries, create execution plans, traverse B-tree indexes, and potentially JOIN multiple tables. The overhead adds up—even a simple SELECT has more work than a hash lookup.

**The memory hierarchy matters**: CPU operations are fastest, memory is ~10x slower, SSD is ~100x slower, and network is ~1000x slower. This is why caching works so well—moving data closer to compute dramatically improves performance.

**Why 86,400?** There are 86,400 seconds in a day (24 × 60 × 60). This constant appears frequently in capacity calculations when converting daily volumes to per-second rates.

```
┌─────────────────────────────────────────────────────────────────┐
│                 LATENCY & QPS REFERENCE                         │
│                                                                 │
│   Operation                    Latency        QPS Capacity      │
│   ─────────────────────────────────────────────────────────     │
│   L1 Cache Reference           0.5 ns         -                 │
│   L2 Cache Reference           7 ns           -                 │
│   Main Memory Reference        100 ns         -                 │
│   SSD Random Read              150 μs         -                 │
│   HDD Seek                     10 ms          -                 │
│   Network Round Trip           500 μs         -                 │
│   ─────────────────────────────────────────────────────────     │
│   MySQL Query                  1-10 ms        ~1,000            │
│   Key-Value Store (Redis)      < 1 ms         ~10,000           │
│   In-Memory Cache              < 0.1 ms       ~100,000 - 1M     │
│                                                                 │
│   Why is Key-Value 10x faster than SQL?                        │
│   • Simple GET/PUT vs query parsing                            │
│   • O(1) hash lookup vs B-tree traversal                       │
│   • No query planning overhead                                 │
│   • No JOINs or complex operations                             │
│                                                                 │
│   Key constant: 86,400 seconds per day (24 × 60 × 60)          │
└─────────────────────────────────────────────────────────────────┘
```

### Estimation Formulas

Back-of-the-envelope calculations help you quickly assess whether a design is feasible. The goal isn't precision—it's understanding the order of magnitude.

**QPS (queries per second)**: Start with **DAU** (daily active users), multiply by average requests per user per day, divide by seconds in a day. Peak traffic is typically 2-3x average.

**Storage**: Multiply records per day by record size. Account for replication (usually 3x for durability) and growth period.

**Bandwidth**: Multiply QPS by average request/response size. Consider both ingress (uploads) and egress (downloads) separately.

**Server Count**: Divide peak QPS by capacity per server. Add buffer for headroom (typically 30-50%).

**Common gotchas**:
- Don't forget replication factors
- Consider read vs write ratios (often 10:1 or 100:1)
- Peak traffic can be 10x+ average for spiky workloads
- Storage grows over time—estimate for retention period

```
┌─────────────────────────────────────────────────────────────────┐
│                 CAPACITY ESTIMATION                             │
│                                                                 │
│   QPS (Queries Per Second)                                      │
│   ─────────────────────────                                     │
│   Average QPS = (DAU × requests_per_user) / 86,400             │
│   Peak QPS = Average × 3 (typical multiplier)                  │
│                                                                 │
│   Example: 1M DAU, 5 requests/user/day                         │
│   Average = (1,000,000 × 5) / 86,400 = ~58 QPS                 │
│   Peak = 58 × 3 = ~174 QPS                                     │
│                                                                 │
│   ─────────────────────────────────────────────────────────     │
│                                                                 │
│   STORAGE                                                       │
│   ───────                                                       │
│   Daily = records_per_day × record_size                        │
│   Total = daily × days × replication_factor                    │
│                                                                 │
│   Example: 100M URLs/day, 500 bytes, 5 years, 3x replication   │
│   Daily = 100M × 500B = 50 GB                                  │
│   5 Years = 50GB × 365 × 5 × 3 = ~275 TB                       │
│                                                                 │
│   ─────────────────────────────────────────────────────────     │
│                                                                 │
│   BANDWIDTH                                                     │
│   ─────────                                                     │
│   BW = QPS × avg_request_size                                  │
│                                                                 │
│   Example: 100 QPS × 10KB = 1 MB/s = 8 Mbps                   │
│                                                                 │
│   ─────────────────────────────────────────────────────────     │
│                                                                 │
│   SERVERS NEEDED                                                │
│   ──────────────                                                │
│   Servers = Peak_QPS / QPS_per_server × 1.5 (buffer)          │
│                                                                 │
│   Example: 3,000 Peak QPS, 500 QPS/server                      │
│   Servers = (3,000 / 500) × 1.5 = 9 servers                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Common Design Examples

### URL Shortener

A URL shortener converts long URLs into short codes and redirects visitors to the original URL. Seems simple, but at scale it's a great example of key-value storage, encoding, and caching.

**Key design decisions**:

1. **Encoding**: Use **Base62** (alphanumeric encoding: characters a-z, A-Z, 0-9) for human-readable codes. With 7 characters, you get 62^7 = 3.5 trillion unique codes—plenty for most use cases.

2. **ID Generation**: Use a distributed ID generator (like Snowflake) to create unique numeric IDs, then encode to Base62. Alternatively, generate random strings and check for collisions.

3. **Storage**: A key-value store (DynamoDB, Redis) is perfect—you're doing simple lookups by short code. For durability, persist to disk with in-memory caching.

4. **Caching**: Most URL access follows power-law distribution—a small percentage of URLs get most traffic. Cache popular URLs in Redis with **TTL** (time-to-live). Expect 80%+ cache hit rate.

5. **Redirection**: Use 301 (permanent) redirects if SEO matters, 302 (temporary) if you want to track clicks. Include analytics logging asynchronously.

```
┌─────────────────────────────────────────────────────────────────┐
│                    URL SHORTENER                                │
│                                                                 │
│   ┌──────┐     ┌─────────────┐     ┌───────────┐               │
│   │Client│────►│Load Balancer│────►│API Servers│               │
│   └──────┘     └─────────────┘     └─────┬─────┘               │
│                                          │                      │
│                          ┌───────────────┴───────────────┐      │
│                          │                               │      │
│                          ▼                               ▼      │
│                    ┌───────────┐                  ┌───────────┐ │
│                    │   Cache   │                  │ Database  │ │
│                    │  (Redis)  │                  │ (Sharded) │ │
│                    └───────────┘                  └───────────┘ │
│                                                                 │
│   WRITE FLOW:                                                   │
│   1. Receive long URL: https://example.com/very/long/path      │
│   2. Generate unique ID (Snowflake or random)                  │
│   3. Encode to Base62: abc1234                                 │
│   4. Store mapping: abc1234 → https://example.com/...          │
│   5. Return: https://short.ly/abc1234                          │
│                                                                 │
│   READ FLOW:                                                    │
│   1. Receive: https://short.ly/abc1234                         │
│   2. Check cache (80%+ hit rate expected)                      │
│   3. If miss, query database                                   │
│   4. 301/302 redirect to original URL                          │
│                                                                 │
│   Encoding: Base62 (a-z, A-Z, 0-9)                             │
│   7 chars = 62^7 = 3.5 trillion unique URLs                    │
└─────────────────────────────────────────────────────────────────┘
```

### Chat Application

A chat application requires real-time bidirectional communication—fundamentally different from request-response HTTP. Key challenges include maintaining persistent connections, delivering messages with low latency, and handling presence/typing indicators.

**WebSocket** is the foundation. Unlike HTTP where the client always initiates, WebSocket maintains a persistent connection allowing the server to push messages to clients instantly. Each connected client maintains a WebSocket to a Connection Manager.

**Connection Manager** tracks which users are connected to which servers. When User A sends a message to User B, the system must find B's server and route the message there. This is often implemented with Redis pub/sub—each server subscribes to channels for its connected users.

**Message persistence** uses a database optimized for time-series data (Cassandra works well). Messages are partitioned by conversation_id so all messages in a conversation are colocated. Support for message ordering, read receipts, and offline delivery adds complexity.

**Presence service** tracks online/offline status and typing indicators. These are ephemeral—no need to persist, but they must propagate quickly. Often implemented with Redis with TTL-based expiration.

```
┌─────────────────────────────────────────────────────────────────┐
│                    CHAT APPLICATION                             │
│                                                                 │
│   ┌──────┐     ┌──────┐                                        │
│   │User A│     │User B│                                        │
│   └──┬───┘     └──┬───┘                                        │
│      │            │                                             │
│      │ WebSocket  │ WebSocket (persistent, bidirectional)      │
│      │            │                                             │
│      ▼            ▼                                             │
│   ┌────────────────────────────┐                               │
│   │    Connection Manager      │  Tracks user ↔ server         │
│   │  (Maintains WS sessions)   │  mapping                       │
│   └────────────┬───────────────┘                               │
│                │                                                │
│                ▼                                                │
│   ┌────────────────────────────┐                               │
│   │      Message Router        │  Redis Pub/Sub for            │
│   │      (Redis Pub/Sub)       │  cross-server routing         │
│   └────────────┬───────────────┘                               │
│                │                                                │
│       ┌────────┴────────┐                                      │
│       ▼                 ▼                                       │
│   ┌───────────┐   ┌───────────┐                                │
│   │ Messages  │   │ Presence  │  Online status,                │
│   │    DB     │   │  Service  │  typing indicators             │
│   │(Cassandra)│   │  (Redis)  │                                │
│   └───────────┘   └───────────┘                                │
│                                                                 │
│   Key Design Decisions:                                         │
│   • WebSocket for real-time bidirectional communication        │
│   • Partition messages by conversation_id                      │
│   • Redis pub/sub for cross-server message routing             │
│   • TTL-based presence with heartbeats                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Important Numbers

```
┌─────────────────────────────────────────────────────────────────┐
│                   IMPORTANT NUMBERS                             │
│                                                                 │
│   Time                           Bytes                          │
│   ────                           ─────                          │
│   1 million seconds ≈ 11.5 days  1 KB  = 1,024 B               │
│   1 billion seconds ≈ 31.7 years 1 MB  = 1,024 KB              │
│   Seconds per day   = 86,400     1 GB  = 1,024 MB              │
│   Seconds per year  ≈ 31.5M      1 TB  = 1,024 GB              │
│                                  1 PB  = 1,024 TB              │
│                                                                 │
│   Network                                                       │
│   ───────                                                       │
│   1 Gbps = 125 MB/s                                            │
│   100 Mbps = 12.5 MB/s                                         │
│                                                                 │
│   Scale                                                         │
│   ─────                                                         │
│   Thousand = 10^3   (3 zeros)                                  │
│   Million  = 10^6   (6 zeros)                                  │
│   Billion  = 10^9   (9 zeros)                                  │
│   Trillion = 10^12  (12 zeros)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### System Design Interview Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                SYSTEM DESIGN CHECKLIST                          │
│                                                                 │
│   1. CLARIFY REQUIREMENTS (5-10 min)                           │
│      □ What are the core features? (functional)                │
│      □ How many users? QPS? Data size? (scale)                 │
│      □ What's the read/write ratio?                            │
│      □ Latency requirements? (real-time vs batch)              │
│      □ Availability target? (99.9%? 99.99%?)                   │
│                                                                 │
│   2. HIGH-LEVEL DESIGN (10-15 min)                             │
│      □ Draw the major components                               │
│      □ Show data flow for key operations                       │
│      □ Identify APIs between components                        │
│                                                                 │
│   3. DEEP DIVE (15-20 min)                                     │
│      □ Database schema design                                  │
│      □ SQL vs NoSQL choice with reasoning                      │
│      □ Caching strategy                                        │
│      □ How to scale each component                             │
│                                                                 │
│   4. ADDRESS BOTTLENECKS (5-10 min)                            │
│      □ Single points of failure?                               │
│      □ What happens when X fails?                              │
│      □ Hot spots or bottlenecks?                               │
│                                                                 │
│   5. WRAP UP (2-5 min)                                         │
│      □ Summarize key design decisions                          │
│      □ Acknowledge trade-offs made                             │
│      □ Mention future improvements                             │
└─────────────────────────────────────────────────────────────────┘
```

### Beyond Pattern Matching: The Interview Mindset

The gap between knowing patterns and actually designing systems that scale comes down to **reasoning about trade-offs in real time**, not memorizing reference architectures.

**The Pattern Trap**: Most candidates can draw a load balancer perfectly, but few can explain when horizontal scaling stops being the answer. Collecting architectural patterns like Pokémon cards—Instagram's feed, Netflix's CDN, Twitter's fanout—isn't mastery. Mastery is understanding *why* each pattern exists and *when* it applies.

**Start With Numbers, Not Boxes**: Senior engineers don't start with architecture diagrams. They start with boring, unglamorous numbers:
- How many users?
- What's the read-to-write ratio?
- What's going to break first when this gets real traffic?

The boxes and arrows come later, after the math basically forces your hand.

```
┌─────────────────────────────────────────────────────────────────┐
│             NUMBERS FIRST: KEY QUESTIONS                        │
│                                                                 │
│   BEFORE drawing any component, answer:                         │
│                                                                 │
│   • DAU (Daily Active Users): current vs. 1yr vs. 5yr?         │
│   • QPS (Queries Per Second): reads? writes? peak?             │
│   • Data size: per record? total? growth rate?                 │
│   • Latency: P50? P99? What's acceptable?                      │
│   • Read/Write ratio: 100:1? 1:1? Write-heavy?                 │
│                                                                 │
│   These numbers DICTATE your architecture choices.              │
│   10K QPS vs 10M QPS = completely different designs.           │
└─────────────────────────────────────────────────────────────────┘
```

**Question Every Default Choice**: When you reach for consistent hashing, ask yourself: "Why consistent hashing here? What problem does it solve that a simple modulo wouldn't?" For a URL shortener with deterministic keys, do you actually need ring-based partitioning, or are you just pattern matching?

```
┌─────────────────────────────────────────────────────────────────┐
│           CHALLENGE YOUR COMPONENT CHOICES                      │
│                                                                 │
│   For EVERY component you add, answer:                          │
│                                                                 │
│   • Why this specific component? (not "because tutorials")     │
│   • What metric proves it's necessary?                         │
│   • What new failure mode does it introduce?                   │
│   • Can the system work without it? If yes, don't add it.     │
│                                                                 │
│   Example: Cache                                                │
│   ✗ "Every system needs caching" (pattern matching)            │
│   ✓ "Cache hit rate ~95% due to power-law access pattern,      │
│      reduces DB load from 50K to 2.5K QPS" (number-driven)     │
│                                                                 │
│   Counter-example: URL shortener with long-tail distribution   │
│   → 40% cache hit rate → caching adds latency + complexity     │
│   → might be better to just scale the database                 │
└─────────────────────────────────────────────────────────────────┘
```

**Failure Mode Thinking**: The best interviewers don't ask "how would you design this"—they ask "what happens when":

- What happens when your primary database goes down mid-transaction?
- What happens when cache invalidation lags by 30 seconds during a viral spike?
- What happens when two datacenters split and both think they're the primary?

~73% of major outages involve state inconsistency during partial failures—the exact scenarios most candidates never rehearse.

```
┌─────────────────────────────────────────────────────────────────┐
│              FAILURE MODE EXERCISE                              │
│                                                                 │
│   For ANY component, ask three failure questions:               │
│                                                                 │
│   DATABASE                                                      │
│   • What if writes succeed but reads lag behind?               │
│   • What if the primary fails during a write?                  │
│   • How do you detect silent corruption?                       │
│                                                                 │
│   CACHE                                                         │
│   • What if eviction is faster than population?                │
│   • What if cache and DB disagree? Which wins?                 │
│   • What's your thundering herd strategy?                      │
│                                                                 │
│   LOAD BALANCER                                                 │
│   • What if health checks pass but service is deadlocked?      │
│   • What if one backend is slow but not failing?               │
│   • How do you handle sticky sessions during failover?         │
│                                                                 │
│   If you can't articulate what breaks and how you'd detect     │
│   it, you're probably not ready.                                │
└─────────────────────────────────────────────────────────────────┘
```

**Start Simple, Evolve With Constraints**: The best design isn't the most sophisticated—it's the simplest thing that could work, with complexity added only when measured constraints force it.

```
┌─────────────────────────────────────────────────────────────────┐
│              EVOLUTION-DRIVEN DESIGN                            │
│                                                                 │
│   START WITH:                                                   │
│   • One database, one server, no cache                         │
│   • Vertical scaling first (it's simpler)                      │
│   • Monolith (until team/scale forces microservices)           │
│                                                                 │
│   ADD COMPLEXITY ONLY WHEN:                                     │
│   • A specific metric crosses a threshold you can NAME          │
│   • You can prove the simpler approach won't work              │
│   • The math forces your hand, not the pattern library         │
│                                                                 │
│   EVERY BOX YOU DRAW should solve a problem you've             │
│   already proven exists.                                        │
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│   │  Simple  │────►│ Measure  │────►│  Evolve  │              │
│   │  Design  │     │ Bottleneck│    │  Targeted │              │
│   └──────────┘     └──────────┘     └──────────┘              │
│        │                                   │                    │
│        └───────────── Repeat ─────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

**Red Flags That Signal Pattern Matching**:

| Red Flag | What It Reveals | Better Approach |
|----------|-----------------|-----------------|
| "We need consistent hashing" | Reaching for patterns before understanding the problem | "Our key distribution is X, so we need Y because..." |
| "Add Redis for caching" | Assuming caching always helps | "Our read/write ratio is X:1, hit rate would be ~Y%" |
| "Use Kafka for messaging" | Pattern matching on queue choice | "We need at-least-once delivery because... Kafka's log compaction helps with..." |
| "Shard the database" | Assuming write scaling is needed | "Current write QPS is X, single-node limit is Y, so we need Z shards" |
| "Add a load balancer" | Reflexive complexity | "We have N servers because each handles X QPS" |

**The Mindset Shift**:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PATTERN MATCHER                             │
│                                                                 │
│   • Starts with architecture diagrams                          │
│   • Adds components "because that's what you do"               │
│   • Can draw systems but can't explain trade-offs              │
│   • Freezes when constraints change                            │
│   • Knows WHAT to build, not WHY                               │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM DESIGNER                              │
│                                                                 │
│   • Starts with numbers and constraints                        │
│   • Adds components when metrics force the decision            │
│   • Can defend every trade-off with data                       │
│   • Adapts when constraints shift                              │
│   • Knows WHY before deciding WHAT                             │
└─────────────────────────────────────────────────────────────────┘
```

**Practice Exercise**: Take any system design you've studied. Pick one component. Remove it. Can the system still work?
- If **yes**: You probably didn't need it.
- If **no**: What metric proves it's necessary?

That's how you build judgment instead of just pattern fluency.

### Trade-off Decision Matrix

```
┌─────────────────────────────────────────────────────────────────┐
│                  TRADE-OFF MATRIX                               │
│                                                                 │
│   Decision          │ Option A         │ Option B              │
│   ──────────────────┼──────────────────┼─────────────────────  │
│   SQL vs NoSQL      │ ACID, JOINs      │ Scale, flexibility    │
│   Push vs Pull      │ Low read latency │ Lower write cost      │
│   Cache vs DB       │ Speed            │ Consistency           │
│   Sync vs Async     │ Consistency      │ Performance           │
│   Monolith vs Micro │ Simplicity       │ Scalability           │
│   Strong vs Event.  │ Correctness      │ Availability          │
│   ──────────────────┼──────────────────┼─────────────────────  │
│   TCP vs UDP        │ Reliability      │ Speed                 │
│   Direct vs VPN     │ Performance      │ Cost, simplicity      │
│   Block vs Object   │ Low latency      │ Scalability, cost     │
│   ──────────────────┼──────────────────┼─────────────────────  │
│                                                                 │
│   There's no universally "right" choice—it depends on your    │
│   specific requirements, constraints, and priorities.          │
└─────────────────────────────────────────────────────────────────┘
```

---

*This guide provides foundational knowledge for system design. Real-world systems combine these patterns based on specific requirements, constraints, and trade-offs. The best design is the simplest one that meets your needs.*

*Last updated: January 2026*
