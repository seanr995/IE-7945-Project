# Agentic taxonomy induction

- model `claude-sonnet-5-5` | request style: Messages API: model, max_tokens, system, messages; JSON text parsed + validated locally; no tool calling; no sampling parameters
- **N_AGENTIC = 80** (no topic count was given to Claude)
- stop reason: refinement round limit (2) reached; unmet: topic-count change <= 1
- coverage 0.988; ambiguity (low-confidence automatic) 0.661; sources {'auto_low_confidence': 1993, 'auto_high_confidence': 825, 'claude': 139, 'propagated_from_claude': 23, 'claude_uncertain': 21, 'non_work_flagged_by_claude': 8, 'novelty_isolated': 6}
- Claude calls: 19 of 20 total; this run 12 normal + 2 reserve; cache hits 0
- skipped decisions: 2

| topic_id | name | members | definition |
|---|---|---:|---|
| AT_0091 | Meeting and Team Coordination | 346 | Organising and facilitating meetings and coordinating day-to-day with specific teams such as procurement or program teams. |
| AT_0086 | General Administrative Support | 283 | Performing generic routine administrative and operational support duties not tied to a specific object, such as mail, leave and request processing. |
| AT_0087 | Order and Solicitation Document Preparation | 218 | Preparing formal documents such as solicitations, specifications, contract correspondence, construction documents and reports. Excludes data entry into records. |
| AT_0098 | Partnership and Relationship Building | 157 | Building and expanding partnerships and customer or prospect relationships. |
| AT_0089 | Facility Maintenance and Local Operations | 145 | Handling upkeep of facilities, supplies and tracking systems and managing local site operations. Excludes generic administrative support and equipment installat |
| AT_0017 | Performance Measurement and Analysis | 113 | Analyzing operational and business data, tracking KPIs and evaluating programs to produce insights. |
| AT_0088 | Equipment Installation, Repair and Operation | 112 | Installing and repairing equipment and performing tests, verification and experiments on equipment and setups. |
| AT_0038 | Stakeholder Liaison and Representation | 108 | Building relationships and communicating with stakeholders, vendors, agencies and partner teams on shared work. |
| AT_0090 | Project and Program Management | 100 | Managing and leading projects and cross-functional initiatives. |
| AT_0001 | Software Architecture and Technical Decision-Making | 83 | Defining software and system architecture, setting technical best practices and making trade-off decisions. Generic 'implement solutions' statements belong in f |
| AT_0030 | Data Entry and Record Documentation | 81 | Entering, abstracting, structuring and preserving information in records and databases. |
| AT_0062 | Engineering Design and Specification | 67 | Designing and specifying technical components, processes and requirements, mainly physical or engineered systems. |
| AT_0002 | Software Feature Development | 66 | Building, implementing and delivering software features and projects within a defined architecture. |
| AT_0040 | Policy, Rulemaking and Regulation | 61 | Developing, revising and implementing policies, rules and protocols, including ensuring adherence to laws and regulations. |
| AT_0103 | Email and Correspondence Communication | 49 | Managing email, correspondence and establishing communication lines. |
| AT_0008 | Strategic and Product Roadmap Planning | 49 | Building, prioritising and scoping product roadmaps and releases, aligned with company strategy. |
| AT_0010 | Sales Prospecting and Deal Closing | 47 | Acquiring new customers by qualifying prospects, running discovery calls, drafting deal documents and closing deals. |
| AT_0012 | Team Building and Supervision | 47 | Building, supervising and developing teams and individuals. |
| AT_0026 | Training, Coaching and Mentoring | 46 | Teaching, coaching or mentoring people to build skills or capacity. |
| AT_0037 | Advisory and Consulting | 45 | Giving expert advice and recommendations on decisions; excludes investment assumption challenge (AT_0052) and policy research. |
| AT_0036 | Legal Case Preparation and Proceedings | 40 | Preparing charges, reviewing briefs, supporting appeals and testifying. |
| AT_0015 | Procurement and Contract Management | 36 | Overseeing sourcing, purchase awards, quotes and job order contracts. |
| AT_0099 | AI Transformation Engagement Delivery | 34 | Leading AI transformation engagements, scaling AI and integrating AI expertise for clients. |
| AT_0052 | Investment Assumption Challenge | 34 | Assessing investment and market opportunities, challenging assumptions and synthesising views and recommendations. |
| AT_0100 | Growth Initiative and Scale-up Driving | 34 | Driving product launches, modernization, alliances and commercial scale-up. |
| AT_0028 | Financial Account Tracking and Reconciliation | 33 | Tracking payments, delinquencies and encumbrances and reconciling entries. |
| AT_0019 | Media and Public Communications | 31 | Engaging journalists and producing information materials for media and the public. |
| AT_0014 | Workflow Mapping and Automation Identification | 28 | Mapping workflows and deciding where to automate or move work to AI agents. |
| AT_0057 | Budget Monitoring and Administration | 27 | Tracking spending, flagging deviations and administering budget changes. |
| AT_0042 | Candidate Sourcing and Hiring Support | 25 | Finding candidates and monitoring the interview process. |
| AT_0104 | Software and Hardware Deployment | 24 | Deploying updates, new software and hardware refreshes. |
| AT_0101 | Venture Incorporation and Lead Conversion | 23 | Leading company incorporation and converting leads. |
| AT_0051 | Strategic Direction Setting | 22 | Defining organisation- or region-level vision, strategy, success measures and structure. |
| AT_0034 | Scheduling and Calendar Management | 22 | Scheduling repairs and rooms and adjusting calendars. |
| AT_0097 | Needs and Eligibility Determination | 21 | Identifying needs, sourcing requirements and making determinations on requests such as accommodations. |
| AT_0093 | Blocker Management | 21 | Monitoring and removing team blockers and managing the backlog. |
| AT_0092 | Bottleneck Identification and Resolution | 21 | Finding and resolving process bottlenecks. |
| AT_0105 | Streaming and Network Infrastructure Engineering | 19 | Designing, building, operating and planning large-scale infrastructure. |
| AT_0009 | Go-to-Market and Launch Planning | 16 | Planning product launches: pricing, packaging, positioning and adoption/ROI messaging. |
| AT_0094 | Model Testing and Validation | 16 | Testing and validating models and experiments and reporting results. |
| AT_0027 | Policy and Environment Monitoring | 15 | Tracking policy changes, officials' activities and markets. |
| AT_0029 | Records and Access Review | 15 | AT_0029: auditing registries, access rights and files for accuracy; AT_0016: reviewing budgets and records to prepare required financial and compliance reports. |
| AT_0060 | Compliance and Audit Management | 13 | Overseeing compliance programs and managing audit processes and system governance; distinct from performing record review. |
| AT_0013 | Fundraising and Sponsorship Development | 11 | Identifying and securing funding from sponsors, grants and investors. |
| AT_0064 | Organizational Leadership | 10 | Leading managers, divisions and organisational resources. |
| AT_0070 | Quality Review and Checks | 10 | AT_0070: quality checks of non-code work outputs; AT_0004: peer review of source code. |
| AT_0044 | Client Case Management and Services | 10 | Managing caseloads and serving clients through intake and follow-up. |
| AT_0018 | Policy and Program Research | 9 | Researching legislation, policy and programs and helping draft amendments. |
| AT_0058 | Accounting Operations Support | 9 | Supporting routine accounting such as close, receivables and balance sheet. |
| AT_0065 | Coaching and Performance Management | 8 | Developing employees and managing performance. |
| AT_0102 | Voter Outreach and Education | 8 | Conducting voter education, outreach and engagement initiatives. |
| AT_0016 | Records, Budget and Compliance Reporting | 8 | Reviewing records and budgets, coding accounts and preparing required reports. |
| AT_0054 | Deal Financing and Structuring | 7 | Structuring and modelling financing for deals and projects. |
| AT_0073 | Product Prioritization | 7 | Balancing user needs, value and constraints, informed by feedback and analytics, to set priorities. |
| AT_0031 | Data Retrieval and Reporting | 6 | AT_0031: pulling data and reporting results; AT_0032: interpreting results, forecasting and testing models. |
| AT_0068 | Road and Grounds Maintenance | 6 | Hands-on upkeep of roads and grounds. |
| AT_0096 | Pricing Determination and Negotiation | 6 | Determining reasonable prices, owning and negotiating pricing. |
| AT_0071 | Data Quality Monitoring and Reporting | 6 | Detecting data inconsistencies and drift in datasets and reporting findings; distinct from general quality review of outputs (AT_0070). |
| AT_0006 | Data and Content Processing | 5 | Migrating content and enriching data, including with LLMs. |
| AT_0095 | Forecast Preparation | 5 | Preparing and updating sales, capital and order forecasts. |
| AT_0069 | Vehicle and Vessel Operation | 5 | Operating vehicles and watercraft. |
| AT_0083 | Metric Definition | 5 | Defining metric contracts and standards. |
| AT_0063 | Engineering Technical Analysis | 4 | Analysing and interpreting technical systems and schematics. |
| AT_0003 | Experience Design | 4 | Designing intuitive user experiences. |
| AT_0080 | Software Debugging | 4 | Diagnosing issues across code, databases, infrastructure and APIs. |
| AT_0053 | Product Teardown Analysis | 4 | Dismantling products or apps to understand how they work or engage users. |
| AT_0075 | Vision Screening and Records | 4 | Screening children's vision and recording results. |
| AT_0004 | Code Review | 3 | Reviewing peers' code for quality and correctness. |
| AT_0079 | Database Optimization | 3 | Tuning databases for performance. |
| AT_0043 | Vendor Screening and Proposal Evaluation | 3 | Screening vendors and evaluating proposals. |
| AT_0059 | Climate Impact Assessment and Reduction | 3 | Assessing and reducing climate and carbon impacts. |
| AT_0067 | Construction Project Oversight | 3 | Directing planning, design and construction projects. |
| AT_0035 | Law Enforcement Operations | 2 | Carrying out enforcement actions such as arrests and extraditions. |
| AT_0082 | Data Integration | 2 | Connecting marketing tools, ad platforms and data sources. |
| AT_0048 | Product Release Gatekeeping | 2 | Controlling quality and timing of product releases. |
| AT_0025 | Field Visits and Home Inspections | 1 | Visiting homes to collect rent and check housekeeping or safety conditions. |
| AT_0050 | Furniture Delivery | 1 | Delivering furniture to destinations. |
| AT_0049 | Copy Editing | 1 | Editing publications for correctness and clarity. |
| AT_0081 | Machine Learning Development | 1 | Applying LLMs to train or supervise models. |
| AT_0076 | Asset Inventory Updating | 1 | Keeping inventory records current in an asset tool. |