// Figure data for the payment-gateway post: sequence diagrams for SequenceSteps.astro and box
// diagrams for BoxDiagram.astro. Message types and field numbers follow ISO 8583 (1987); the
// amounts follow the post's running example, a €42.00 payment by merchant acct_1042.

import type { BoxEdge, BoxNode } from "./boxDiagram";
import type { Message, Participant } from "./sequence";

// --- Authorization ---------------------------------------------------------------------------

export const cardParticipants: Participant[] = [
	{ id: "ch", label: "Cardholder" },
	{ id: "m", label: "Merchant" },
	{ id: "gw", label: "Gateway", sub: "this system" },
	{ id: "acq", label: "Acquirer", sub: "merchant's bank" },
	{ id: "net", label: "Card network", sub: "Visa / Mastercard" },
	{ id: "iss", label: "Issuer", sub: "cardholder's bank" },
];

export const authMessages: Message[] = [
	{
		from: "ch",
		to: "m",
		label: "pay €42.00",
		step: "Checkout",
		note: "The cardholder types card details into the checkout page. The card fields are served by the gateway, so the card number goes straight to the gateway's vault and the merchant only gets back a token.",
	},
	{
		from: "m",
		to: "gw",
		label: "POST /payments, Idempotency-Key",
		step: "API call",
		note: "The merchant's server calls the gateway: amount <code>4200</code>, currency <code>EUR</code>, the card token and an <strong>idempotency key</strong> it generated for this checkout.",
	},
	{
		from: "gw",
		to: "gw",
		label: "INSERT payment (created)",
		step: "Record",
		note: "Before talking to anyone outside, the gateway writes the payment to MySQL in state <code>created</code>. If it crashes later, the row says a request may be in flight.",
	},
	{
		from: "gw",
		to: "acq",
		label: "authorize 4200 EUR",
		step: "To acquirer",
		note: "The gateway picks an acquirer (by card brand, currency, cost and health) and sends it the authorization.",
	},
	{
		from: "acq",
		to: "net",
		label: "0100",
		step: "0100",
		note: "The acquirer sends an ISO 8583 message of type <code>0100</code> (authorization request): field 2 the card number, field 4 the amount <code>000000004200</code>, field 49 the currency <code>978</code> (euro), field 11 a trace number, fields 41–42 the terminal and merchant.",
	},
	{
		from: "net",
		to: "iss",
		label: "0100",
		step: "To issuer",
		note: "The network reads the card number's leading digits (the BIN) to find the issuing bank and forwards the request. If the issuer does not answer in time, Visa's Stand-In Processing can answer for it using limits the issuer set.",
	},
	{
		from: "iss",
		to: "iss",
		label: "check funds, fraud; hold €42",
		step: "Decide",
		note: "The issuer checks the card's status, available credit and its fraud models, then places a <strong>hold</strong> of €42.00. No money has moved: the cardholder's available balance just drops.",
	},
	{
		from: "iss",
		to: "net",
		label: "0110 · 39=00 · 38=A1B2C3",
		step: "0110",
		reply: true,
		tone: 1,
		note: "The reply has type <code>0110</code>. Field 39, the response code, is <code>00</code> (approved); a decline would carry for example <code>05</code> (do not honor) or <code>51</code> (insufficient funds). Field 38 carries the six-character authorization code.",
	},
	{
		from: "net",
		to: "acq",
		label: "0110 approved",
		step: "Back",
		reply: true,
		tone: 1,
		note: "The network passes the response back along the same path.",
	},
	{
		from: "acq",
		to: "gw",
		label: "approved, A1B2C3",
		step: "To gateway",
		reply: true,
		tone: 1,
		note: "The acquirer returns the result and the network's reference numbers, which the gateway stores: clearing and dispute files arrive later keyed by them.",
	},
	{
		from: "gw",
		to: "gw",
		label: "UPDATE status = authorized",
		step: "Update",
		note: "One MySQL transaction moves the payment to <code>authorized</code>, saves the response for the idempotency key and adds an event to the outbox.",
	},
	{
		from: "gw",
		to: "m",
		label: "201 authorized",
		step: "Reply",
		reply: true,
		tone: 1,
		note: "The merchant shows the order confirmation. The whole round trip typically takes on the order of a second.",
	},
];

// --- Capture, clearing, settlement -----------------------------------------------------------

export const settleMessages: Message[] = [
	{
		from: "m",
		to: "gw",
		label: "POST /payments/{id}/capture",
		step: "Capture",
		phase: "Capture: when the order ships",
		note: "Capturing tells the gateway to collect the authorized amount (or less). Many merchants capture right away; shops that ship later capture at shipment.",
	},
	{
		from: "gw",
		to: "gw",
		label: "status = captured; ledger rows",
		step: "Ledger",
		note: "One transaction moves the payment to <code>captured</code> and writes its ledger entries: the gateway now expects €42.00 from the acquirer and owes the merchant €42.00 less its fee.",
	},
	{
		from: "gw",
		to: "acq",
		label: "captures for the day",
		step: "Batch",
		phase: "Clearing: batch, usually daily",
		note: "Captured payments go to the acquirer in a batch.",
	},
	{
		from: "acq",
		to: "net",
		label: "clearing records",
		step: "Clearing",
		note: "The acquirer submits clearing records: Visa clears through BASE II, Mastercard through its Global Clearing Management System (GCMS). The network works out interchange (the fee paid to the issuer) and what each bank owes.",
	},
	{
		from: "net",
		to: "iss",
		label: "clearing records",
		step: "To issuer",
		note: "The issuer receives the clearing record and matches it to the earlier hold.",
	},
	{
		from: "iss",
		to: "ch",
		label: "post to statement",
		step: "Statement",
		note: "The €42.00 now appears as a posted transaction on the cardholder's statement instead of a pending hold.",
	},
	{
		from: "iss",
		to: "net",
		label: "net funds",
		step: "Issuer pays",
		phase: "Settlement: net positions, once a day",
		tone: 3,
		note: "Banks do not pay each other per transaction. The network nets every position for the day and each issuer sends one amount: what its cardholders spent, less the interchange it earns.",
	},
	{
		from: "net",
		to: "acq",
		label: "net funds",
		step: "Network pays",
		tone: 3,
		note: "The network pays each acquirer its net amount, less network fees.",
	},
	{
		from: "acq",
		to: "gw",
		label: "funds + settlement file",
		step: "Funds",
		tone: 3,
		note: "The acquirer pays the gateway and sends a settlement file that lists every transaction and fee.",
	},
	{
		from: "gw",
		to: "gw",
		label: "reconcile file ↔ ledger",
		step: "Reconcile",
		note: "The gateway matches every line of the file to a payment in its ledger. Anything that does not match (a missing capture, a wrong fee) becomes a break for someone to investigate.",
	},
	{
		from: "gw",
		to: "m",
		label: "payout €41.12",
		step: "Payout",
		tone: 3,
		note: "Finally the merchant receives its money, less the gateway's fee. This is days after the cardholder tapped their card.",
	},
];

// --- Idempotent retry ------------------------------------------------------------------------

export const retryParticipants: Participant[] = [
	{ id: "m", label: "Merchant" },
	{ id: "api", label: "Gateway API" },
	{ id: "db", label: "MySQL", sub: "merchant's shard" },
	{ id: "acq", label: "Acquirer" },
];

export const retryMessages: Message[] = [
	{
		from: "m",
		to: "api",
		label: "POST /payments, key k7",
		step: "Request",
		note: "The merchant sends a payment with idempotency key <code>k7</code>.",
	},
	{
		from: "api",
		to: "db",
		label: "INSERT idempotency_keys (k7)",
		step: "Claim key",
		note: "The API inserts <code>(merchant_id, k7)</code> into a table with a unique key on those two columns. The insert succeeds, so this request owns the key. The row records a hash of the request body and no response yet.",
	},
	{
		from: "api",
		to: "acq",
		label: "authorize",
		step: "Authorize",
		note: "Only the request that owns the key talks to the acquirer.",
	},
	{
		from: "acq",
		to: "api",
		label: "approved",
		step: "Approved",
		reply: true,
		tone: 1,
		note: "The issuer approves.",
	},
	{
		from: "api",
		to: "db",
		label: "UPDATE payment; save response",
		step: "Save",
		note: "One transaction updates the payment and stores the response body on the key row.",
	},
	{
		from: "api",
		to: "m",
		label: "201 authorized",
		step: "Lost",
		reply: true,
		lost: true,
		tone: 2,
		note: "The response never arrives: the merchant's connection times out. The merchant cannot tell whether the payment happened.",
	},
	{
		from: "m",
		to: "api",
		label: "retry: POST /payments, key k7",
		step: "Retry",
		note: "So it sends the same request again with the same key, which is safe only because of the key.",
	},
	{
		from: "api",
		to: "db",
		label: "INSERT k7 → duplicate key",
		step: "Duplicate",
		tone: 2,
		note: "The insert fails with a duplicate-key error (MySQL error 1062). The API reads the existing row instead. If its response were still empty, the first request would still be running and the API would answer <code>409</code>.",
	},
	{
		from: "db",
		to: "api",
		label: "stored response",
		step: "Stored",
		reply: true,
		note: "The row holds the saved response. The API also checks that the request hash matches, to catch a key reused for a different payment.",
	},
	{
		from: "api",
		to: "m",
		label: "201 authorized (same body)",
		step: "Replay",
		reply: true,
		tone: 1,
		note: "The merchant gets exactly the first response. The card was authorized once.",
	},
];

// --- Online resharding -----------------------------------------------------------------------

export const reshardParticipants: Participant[] = [
	{ id: "router", label: "Shard router" },
	{ id: "src", label: "Shard 0", sub: "source" },
	{ id: "job", label: "Reshard job" },
	{ id: "dst", label: "Shard 4", sub: "new" },
];

export const reshardMessages: Message[] = [
	{
		from: "job",
		to: "src",
		label: "consistent snapshot of bucket 3",
		step: "Snapshot",
		phase: "Copy: hours, with live traffic",
		note: "Going from 4 to 5 shards, shard 4 takes bucket 3 from shard 0 (and one bucket each from shards 1 and 2). The job reads bucket 3's rows from a consistent snapshot and notes the binlog position (GTID set) the snapshot corresponds to.",
	},
	{
		from: "job",
		to: "dst",
		label: "bulk INSERT rows",
		step: "Copy",
		note: "It copies the rows to shard 4. Meanwhile shard 0 keeps serving reads and writes for bucket 3.",
	},
	{
		from: "src",
		to: "job",
		label: "binlog events after snapshot",
		step: "Stream",
		phase: "Catch up",
		reply: true,
		note: "The job tails shard 0's binary log from the snapshot's position and keeps the events for rows in bucket 3.",
	},
	{
		from: "job",
		to: "dst",
		label: "apply events",
		step: "Apply",
		note: "Applying them brings shard 4 to within seconds of shard 0.",
	},
	{
		from: "job",
		to: "job",
		label: "diff source vs target",
		step: "Verify",
		note: "Before switching, compare the two copies row by row (Vitess calls this VDiff). For money, this check is not optional.",
	},
	{
		from: "router",
		to: "src",
		label: "stop writes to bucket 3",
		step: "Freeze",
		phase: "Cut over: seconds",
		tone: 2,
		note: "The router holds new writes for bucket 3. Writes for every other bucket continue.",
	},
	{
		from: "job",
		to: "dst",
		label: "apply last events; lag = 0",
		step: "Drain",
		note: "The job applies the final events, so shard 4 now has every committed write for bucket 3.",
	},
	{
		from: "router",
		to: "router",
		label: "directory: bucket 3 → shard 4",
		step: "Switch",
		note: "One update to the directory, which every router instance watches, moves the bucket.",
	},
	{
		from: "dst",
		to: "src",
		label: "reverse replication",
		step: "Reverse",
		reply: true,
		note: "Shard 4's changes stream back to shard 0, so switching back is possible if something is wrong.",
	},
	{
		from: "router",
		to: "dst",
		label: "writes resume on shard 4",
		step: "Resume",
		tone: 1,
		note: "Held writes are released to shard 4. Cash App reported less than a second of downtime for its first split done this way.",
	},
];

// --- Cross-shard transfer with an outbox ---------------------------------------------------

export const transferParticipants: Participant[] = [
	{ id: "api", label: "Gateway API" },
	{ id: "a", label: "Shard 1", sub: "platform acct_2001" },
	{ id: "relay", label: "Outbox relay" },
	{ id: "b", label: "Shard 3", sub: "seller acct_3107" },
];

export const transferMessages: Message[] = [
	{
		from: "api",
		to: "a",
		label: "debit €30; INSERT outbox (t_9)",
		step: "Debit",
		note: "A marketplace moves €30.00 from its balance to a seller whose account lives on another shard. One local transaction on shard 1 debits the platform and writes an outbox row <code>t_9</code> that says “credit acct_3107 €30.00”. Money is now in a ledger account for transfers in transit.",
	},
	{
		from: "a",
		to: "relay",
		label: "outbox row t_9",
		step: "Read",
		reply: true,
		note: "A relay process reads unsent outbox rows from every shard, in order.",
	},
	{
		from: "relay",
		to: "b",
		label: "INSERT applied (t_9); credit €30",
		step: "Credit",
		note: "On shard 3, one transaction inserts <code>t_9</code> into a table of applied transfers (unique key) and credits the seller.",
	},
	{
		from: "b",
		to: "relay",
		label: "ok",
		step: "Crash",
		reply: true,
		lost: true,
		tone: 2,
		note: "The relay crashes before it sees the reply. After restart it does not know whether the credit happened.",
	},
	{
		from: "relay",
		to: "b",
		label: "retry t_9 → duplicate key",
		step: "Retry",
		note: "It sends <code>t_9</code> again. The insert into the applied-transfers table fails with a duplicate key, so the transaction rolls back and the seller is not credited twice.",
	},
	{
		from: "b",
		to: "relay",
		label: "already applied",
		step: "Ack",
		reply: true,
		tone: 1,
		note: "The relay treats the duplicate as success.",
	},
	{
		from: "relay",
		to: "a",
		label: "mark t_9 sent",
		step: "Done",
		note: "The outbox row is marked sent. The transfer is complete on both shards, without a transaction that spans them.",
	},
];

// --- Gateway architecture --------------------------------------------------------------------

export const gatewayNodes: BoxNode[] = [
	{
		id: "merchant",
		label: "Merchant",
		sub: "server + checkout",
		col: 0,
		row: 1,
		group: 0,
		note: "Calls the API to create, capture and refund payments, and receives webhooks when their state changes.",
	},
	{
		id: "vault",
		label: "Card vault",
		sub: "PCI DSS zone",
		col: 1,
		row: 0,
		group: 4,
		note: "Stores card numbers encrypted and hands out tokens. Keeping card numbers in one small, locked-down service keeps the rest of the system out of PCI DSS audit scope.",
	},
	{
		id: "api",
		label: "Payments API",
		sub: "auth, idempotency",
		col: 1,
		row: 1,
		group: 1,
		note: "Authenticates the merchant, validates the request and enforces idempotency keys before anything else happens.",
	},
	{
		id: "risk",
		label: "Risk engine",
		sub: "fraud scoring",
		col: 2,
		row: 0,
		group: 5,
		note: "Scores each payment from the card, device and merchant history, and can block it or ask for 3-D Secure authentication.",
	},
	{
		id: "svc",
		label: "Payment service",
		sub: "state machine",
		col: 2,
		row: 1,
		group: 1,
		note: "Owns the payment's state machine. Every transition is a MySQL transaction that changes the payment, writes ledger entries and adds an outbox event together.",
	},
	{
		id: "router",
		label: "Acquirer router",
		sub: "connectors",
		col: 3,
		row: 1,
		group: 2,
		note: "Translates a payment into each acquirer's protocol and picks an acquirer by card brand, region, cost and current error rate.",
	},
	{
		id: "acq",
		label: "Acquirers",
		sub: "→ Visa / Mastercard",
		col: 4,
		row: 1,
		group: 0,
		note: "The banks that connect the gateway to the card networks. They also send daily settlement files.",
	},
	{
		id: "recon",
		label: "Reconciliation",
		sub: "settlement files",
		col: 4,
		row: 0,
		group: 2,
		note: "Matches each line of the acquirers' settlement files against the ledger and reports breaks.",
	},
	{
		id: "ledger",
		label: "Ledger",
		sub: "double entry",
		col: 1,
		row: 2,
		group: 3,
		note: "The record of who owes whom. Every money movement is a set of entries that sum to zero, written in the same transaction as the state change that caused it.",
	},
	{
		id: "db",
		label: "MySQL shards",
		sub: "payments, ledger",
		col: 2,
		row: 2,
		group: 3,
		note: "The system of record: payments, ledger entries, idempotency keys and the outbox, sharded by merchant.",
	},
	{
		id: "relay",
		label: "Outbox relay",
		col: 3,
		row: 2,
		group: 2,
		note: "Reads committed outbox rows and publishes them, so an event is sent if and only if its transaction committed.",
	},
	{
		id: "hooks",
		label: "Webhooks",
		sub: "to merchant",
		col: 4,
		row: 2,
		group: 0,
		note: "Delivers events such as <code>payment.captured</code> to the merchant, retrying until acknowledged.",
	},
];

export const gatewayEdges: BoxEdge[] = [
	{ from: "merchant", to: "api" },
	{ from: "api", to: "vault", label: "token" },
	{ from: "api", to: "svc" },
	{ from: "svc", to: "risk" },
	{ from: "svc", to: "router" },
	{ from: "router", to: "acq", both: true },
	{ from: "acq", to: "recon", dashed: true },
	{ from: "svc", to: "db" },
	{ from: "svc", to: "ledger" },
	{ from: "ledger", to: "db" },
	{ from: "db", to: "relay" },
	{ from: "relay", to: "hooks" },
];

// --- Payment state machine ---------------------------------------------------------------------

export const stateNodes: BoxNode[] = [
	{
		id: "created",
		label: "created",
		col: 0,
		row: 0,
		group: 0,
		note: "Written before any external call. A payment stuck here after a crash is checked against the acquirer before anything else happens.",
	},
	{
		id: "authorized",
		label: "authorized",
		col: 1,
		row: 0,
		group: 1,
		note: "The issuer holds the funds. The merchant can capture or void. Holds expire if never captured.",
	},
	{
		id: "captured",
		label: "captured",
		col: 2,
		row: 0,
		group: 1,
		note: "The merchant asked for the money. Ledger entries are written: receivable from the acquirer, payable to the merchant, fee.",
	},
	{
		id: "settled",
		label: "settled",
		col: 3,
		row: 0,
		group: 3,
		note: "The acquirer's funds arrived and reconciliation matched this payment.",
	},
	{
		id: "failed",
		label: "failed",
		col: 0,
		row: 1,
		group: 4,
		note: "Declined by the issuer or blocked by risk. Terminal.",
	},
	{
		id: "voided",
		label: "voided",
		col: 1,
		row: 1,
		group: 0,
		note: "The merchant released the hold before capture (an <code>0400</code> reversal goes to the issuer). Terminal.",
	},
	{
		id: "refunded",
		label: "refunded",
		col: 2,
		row: 1,
		group: 2,
		note: "Money returned to the cardholder, in full or in part. Several partial refunds can add up to the captured amount, never more.",
	},
	{
		id: "disputed",
		label: "disputed",
		col: 3,
		row: 1,
		group: 4,
		note: "The cardholder disputed the charge with the issuer (a chargeback). The money is pulled back until the dispute is resolved.",
	},
];

export const stateEdges: BoxEdge[] = [
	{ from: "created", to: "authorized", label: "00" },
	{ from: "created", to: "failed", label: "decline" },
	{ from: "authorized", to: "captured", label: "capture" },
	{ from: "authorized", to: "voided", label: "void" },
	{ from: "captured", to: "settled", label: "funds" },
	{ from: "captured", to: "refunded", label: "refund" },
	{ from: "settled", to: "refunded" },
	{ from: "settled", to: "disputed", label: "chargeback" },
];

// --- Sharded deployment ----------------------------------------------------------------------

const shardBoxes = (k: number, col: number): BoxNode[] => [
	{ id: `c${k}`, label: `shard ${k}`, col, row: 2, w: 2, h: 2, group: -1 },
	{
		id: `p${k}`,
		label: "primary",
		sub: "zone a",
		col,
		row: 2,
		w: 2,
		group: 1,
		note: `Shard ${k}'s primary takes every write for its buckets. A commit returns only after the redo log and binary log are flushed and at least one replica has the transaction on disk.`,
	},
	{
		id: `r${k}b`,
		label: "replica",
		sub: "zone b",
		col,
		row: 3,
		group: 0,
		note: "Semi-synchronous replica in another availability zone: it acknowledged every committed transaction, so it can be promoted without losing one.",
	},
	{
		id: `r${k}c`,
		label: "replica",
		sub: "zone c",
		col: col + 1,
		row: 3,
		group: 0,
		note: "A second replica, for reads that can be a little stale (reports, dashboards) and as another failover candidate.",
	},
];

export const shardedNodes: BoxNode[] = [
	{
		id: "svc",
		label: "Payment service",
		col: 2,
		row: 0,
		w: 2,
		group: 1,
		note: "Sends SQL with the merchant's ID (or a payment ID, which carries its bucket) and never needs to know which shard it hits.",
	},
	{
		id: "router",
		label: "Shard router",
		sub: "hash → bucket → shard",
		col: 2,
		row: 1,
		w: 2,
		group: 2,
		note: "Hashes the shard key to one of 4,096 buckets and looks the bucket up in the directory. Vitess's VTGate plays this role.",
	},
	{
		id: "topo",
		label: "Directory",
		sub: "etcd / ZooKeeper",
		col: 5,
		row: 1,
		group: 5,
		note: "The bucket → shard map, kept in a small consistent store that every router watches. Resharding changes only this map.",
	},
	...shardBoxes(0, 0),
	...shardBoxes(1, 2),
	...shardBoxes(2, 4),
];

export const shardedEdges: BoxEdge[] = [
	{ from: "svc", to: "router" },
	{ from: "router", to: "topo", dashed: true },
	...[0, 1, 2].flatMap((k) => [
		{ from: "router", to: `p${k}` },
		{ from: `p${k}`, to: `r${k}b` },
		{ from: `p${k}`, to: `r${k}c` },
	]),
];

// --- NDB Cluster -------------------------------------------------------------------------------

export const ndbNodes: BoxNode[] = [
	{
		id: "app",
		label: "Application",
		col: 1,
		row: 0,
		w: 2,
		group: 0,
		note: "Connects to any SQL node with an ordinary MySQL client.",
	},
	{
		id: "mgm",
		label: "Management node",
		sub: "ndb_mgmd",
		col: 4,
		row: 0,
		group: 5,
		note: "Holds the cluster configuration and arbitrates when nodes fail. It is not on the data path.",
	},
	{
		id: "sql1",
		label: "SQL node",
		sub: "mysqld",
		col: 0,
		row: 1,
		w: 2,
		group: 1,
		note: "A normal <code>mysqld</code> that parses SQL and stores nothing itself: tables with <code>ENGINE=NDB</code> live on the data nodes, so every SQL node sees the same data.",
	},
	{
		id: "sql2",
		label: "SQL node",
		sub: "mysqld",
		col: 2,
		row: 1,
		w: 2,
		group: 1,
		note: "Add SQL nodes to scale query parsing; add data nodes to scale storage and writes.",
	},
	{ id: "ng0", label: "node group 0", col: 0, row: 2, w: 2, group: -1 },
	{
		id: "d1",
		label: "data node 1",
		sub: "partitions 0, 2",
		col: 0,
		row: 2,
		group: 3,
		note: "Each data node keeps a copy (a fragment replica) of every partition assigned to its node group. With <code>NoOfReplicas=2</code>, node groups = data nodes / 2.",
	},
	{
		id: "d2",
		label: "data node 2",
		sub: "partitions 0, 2",
		col: 1,
		row: 2,
		group: 3,
		note: "The other replica in node group 0. Writes are applied to both replicas synchronously before commit, so either can take over in under a second.",
	},
	{ id: "ng1", label: "node group 1", col: 2, row: 2, w: 2, group: -1 },
	{
		id: "d3",
		label: "data node 3",
		sub: "partitions 1, 3",
		col: 2,
		row: 2,
		group: 3,
		note: "Node group 1 holds the other half of the rows. NDB hashes each row's primary key to pick its partition, so this is sharding done inside the storage engine.",
	},
	{
		id: "d4",
		label: "data node 4",
		sub: "partitions 1, 3",
		col: 3,
		row: 2,
		group: 3,
		note: "Adding a node group online adds capacity; existing rows can then be reorganized onto it.",
	},
];

export const ndbEdges: BoxEdge[] = [
	{ from: "app", to: "sql1" },
	{ from: "app", to: "sql2" },
	{ from: "sql1", to: "d1" },
	{ from: "sql1", to: "d3" },
	{ from: "sql2", to: "d2" },
	{ from: "sql2", to: "d4" },
	{ from: "d1", to: "d2", both: true, dashed: true },
	{ from: "d3", to: "d4", both: true, dashed: true },
	{ from: "mgm", to: "sql2", dashed: true },
];

// --- A fractured read --------------------------------------------------------------------------

export const fracturedParticipants: Participant[] = [
	{ id: "tx", label: "Transfer", sub: "atomic, 2PC" },
	{ id: "a", label: "Shard 1", sub: "platform: €100" },
	{ id: "b", label: "Shard 3", sub: "seller: €0" },
	{ id: "audit", label: "Balance check", sub: "SUM across shards" },
];

export const fracturedMessages: Message[] = [
	{
		from: "tx",
		to: "a",
		label: "PREPARE: platform −30",
		step: "Prepare 1",
		note: "One distributed transaction moves €30 from the platform (shard 1) to the seller (shard 3). Both shards prepare: the change is durable and locked, but not yet visible.",
	},
	{
		from: "tx",
		to: "b",
		label: "PREPARE: seller +30",
		step: "Prepare 2",
		note: "Both shards have voted yes, so the transaction <em>will</em> commit on both. Atomicity is guaranteed.",
	},
	{
		from: "tx",
		to: "a",
		label: "COMMIT",
		step: "Commit 1",
		tone: 1,
		note: "The coordinator sends the commits. Shard 1 applies its commit first: the platform's balance is now €70 to any new reader.",
	},
	{
		from: "audit",
		to: "a",
		label: "read platform → €70",
		step: "Read 1",
		phase: "A reader runs in between",
		note: "A job that checks the invariant “total money is €100” starts now and reads shard 1: €70.",
	},
	{
		from: "audit",
		to: "b",
		label: "read seller → €0",
		step: "Read 2",
		tone: 2,
		note: "It reads shard 3, which has not applied its commit yet: €0. The job sees €70 + €0 = €70, money that never existed, and raises a false alarm (or worse, a report or payout is computed from it).",
	},
	{
		from: "tx",
		to: "b",
		label: "COMMIT",
		step: "Commit 2",
		tone: 1,
		note: "A moment later shard 3 commits and the total is €100 again. Nothing was lost; the reader saw a state that never existed as a whole. This is a <strong>fractured read</strong>: atomic commit without isolation across shards.",
	},
];

// --- Two-phase commit over consensus groups -----------------------------------------------------

export const consensusParticipants: Participant[] = [
	{ id: "api", label: "Gateway API" },
	{ id: "a", label: "Range A leader", sub: "platform acct_2001" },
	{ id: "af", label: "Range A followers", sub: "2 other zones" },
	{ id: "b", label: "Range B leader", sub: "seller acct_3107" },
	{ id: "bf", label: "Range B followers", sub: "2 other zones" },
];

export const consensusMessages: Message[] = [
	{
		from: "api",
		to: "a",
		label: "debit €30 (lock)",
		step: "Write A",
		note: "The same transfer as before, now as one transaction in a distributed SQL database. Each <strong>range</strong> (a slice of the key space) is a consensus group: a leader and followers in other zones. The write to range A takes a lock and is held provisionally.",
	},
	{
		from: "api",
		to: "b",
		label: "credit €30 (lock)",
		step: "Write B",
		note: "The write to range B does the same on a different group, possibly on different servers.",
	},
	{
		from: "api",
		to: "a",
		label: "COMMIT",
		step: "Commit",
		note: "The client commits. Range A's leader acts as the coordinator (Spanner calls it the coordinator leader).",
	},
	{
		from: "b",
		to: "bf",
		label: "replicate PREPARE",
		step: "Prepare B",
		phase: "Phase 1: prepare, each through consensus",
		note: "Range B's leader writes a prepare record through consensus: it is durable once a majority of B's replicas have it on disk.",
	},
	{
		from: "b",
		to: "a",
		label: "prepared",
		step: "Vote",
		reply: true,
		tone: 1,
		note: "B votes yes to the coordinator.",
	},
	{
		from: "a",
		to: "af",
		label: "replicate COMMIT record",
		step: "Decide",
		phase: "Phase 2: the decision, through consensus",
		note: "The coordinator writes the commit decision through its own consensus group. Unlike MySQL XA, the coordinator's state is replicated, so the crash of one server cannot leave the transaction stuck: a new leader of range A knows the outcome.",
	},
	{
		from: "a",
		to: "api",
		label: "committed",
		step: "Ack",
		reply: true,
		tone: 1,
		note: "The client gets its answer. (Spanner first waits out its clock uncertainty, “commit wait”, so that commit timestamps respect real time.)",
	},
	{
		from: "a",
		to: "b",
		label: "commit at timestamp t",
		step: "Apply",
		note: "Participants apply the commit. Readers use the commit timestamp: a snapshot read at any time either sees both the debit and the credit or neither, so the fractured read from the previous figure cannot happen.",
	},
];
