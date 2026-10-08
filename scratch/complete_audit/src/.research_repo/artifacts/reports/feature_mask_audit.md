# Feature Mask Audit

| Feature Name | Eligible | Category | Agrees with `continuous_numerical`? | Evidence | Ambiguity |
|---|---|---|---|---|---|
| ACK Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| Active Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Active Mean | True | continuous measurement | True | Statistical aggregation |  |
| Active Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Active Std | True | continuous measurement | True | Statistical aggregation |  |
| Bwd Blk Rate Avg | True | continuous measurement | True | Derived rate field |  |
| Bwd Byts/b Avg | True | continuous measurement | True | Statistical aggregation |  |
| Bwd Header Len | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd IAT Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd IAT Mean | True | continuous measurement | True | Statistical aggregation |  |
| Bwd IAT Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd IAT Std | True | continuous measurement | True | Statistical aggregation |  |
| Bwd IAT Tot | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd PSH Flags | False | binary flag | True | Networking concept (TCP flags) |  |
| Bwd Pkt Len Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd Pkt Len Mean | True | continuous measurement | True | Statistical aggregation |  |
| Bwd Pkt Len Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Bwd Pkt Len Std | True | continuous measurement | True | Statistical aggregation |  |
| Bwd Pkts/b Avg | True | continuous measurement | True | Statistical aggregation |  |
| Bwd Pkts/s | True | continuous measurement | True | Derived rate field |  |
| Bwd Seg Size Avg | True | continuous measurement | True | Statistical aggregation |  |
| Bwd URG Flags | False | binary flag | True | Networking concept (TCP flags) |  |
| CWE Flag Count | False | binary flag | True | Networking concept (TCP flags) |  |
| Down/Up Ratio | False | ambiguous | True | No authoritative schema document found |  |
| Dst Port | False | categorical code | True | Networking standard |  |
| ECE Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| FIN Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| Flow Byts/s | True | continuous measurement | True | Derived rate field |  |
| Flow Duration | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Flow IAT Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Flow IAT Mean | True | continuous measurement | True | Statistical aggregation |  |
| Flow IAT Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Flow IAT Std | True | continuous measurement | True | Statistical aggregation |  |
| Flow Pkts/s | True | continuous measurement | True | Derived rate field |  |
| Fwd Act Data Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Fwd Blk Rate Avg | True | continuous measurement | True | Derived rate field |  |
| Fwd Byts/b Avg | True | continuous measurement | True | Statistical aggregation |  |
| Fwd Header Len | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd IAT Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd IAT Mean | True | continuous measurement | True | Statistical aggregation |  |
| Fwd IAT Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd IAT Std | True | continuous measurement | True | Statistical aggregation |  |
| Fwd IAT Tot | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd PSH Flags | False | binary flag | True | Networking concept (TCP flags) |  |
| Fwd Pkt Len Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd Pkt Len Mean | True | continuous measurement | True | Statistical aggregation |  |
| Fwd Pkt Len Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd Pkt Len Std | True | continuous measurement | True | Statistical aggregation |  |
| Fwd Pkts/b Avg | True | continuous measurement | True | Statistical aggregation |  |
| Fwd Pkts/s | True | continuous measurement | True | Derived rate field |  |
| Fwd Seg Size Avg | True | continuous measurement | True | Statistical aggregation |  |
| Fwd Seg Size Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Fwd URG Flags | False | binary flag | True | Networking concept (TCP flags) |  |
| Idle Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Idle Mean | True | continuous measurement | True | Statistical aggregation |  |
| Idle Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Idle Std | True | continuous measurement | True | Statistical aggregation |  |
| Init Bwd Win Byts | True | integer count | False | Byte count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Init Fwd Win Byts | True | integer count | False | Byte count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| PSH Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| Pkt Len Max | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Pkt Len Mean | True | continuous measurement | True | Statistical aggregation |  |
| Pkt Len Min | True | ambiguous | False | No authoritative schema document found | Feature is ambiguous but is marked eligible under 'continuous_numerical' rules. |
| Pkt Len Std | True | continuous measurement | True | Statistical aggregation |  |
| Pkt Len Var | True | continuous measurement | True | Statistical aggregation |  |
| Pkt Size Avg | True | continuous measurement | True | Statistical aggregation |  |
| Protocol | False | categorical code | True | Networking standard |  |
| RST Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| SYN Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
| Subflow Bwd Byts | True | integer count | False | Byte count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Subflow Bwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Subflow Fwd Byts | True | integer count | False | Byte count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Subflow Fwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Tot Bwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| Tot Fwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| TotLen Bwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| TotLen Fwd Pkts | True | integer count | False | Packet count | Feature is integer count but is marked eligible under 'continuous_numerical' rules. |
| URG Flag Cnt | False | binary flag | True | Networking concept (TCP flags) |  |
