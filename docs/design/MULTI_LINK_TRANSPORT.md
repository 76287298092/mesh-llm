# Wired + wireless multi-link transport

Status: design. Nothing here is implemented; the section "Falsification gate"
is the measurement that decides whether it should be.

## What exists today

- Transport is **iroh 1.0.3**: QUIC with hole punching, relay fallback, and
  `portmapper`. One path per connection — iroh has no MP-QUIC, and the IETF
  multipath draft is not in it.
- The endpoint binds `0.0.0.0:0`, so it already listens on every interface, and
  `network_diagnostics` already gathers `lan_candidates: Vec<SocketAddr>`.
- Path state per peer already exists and is already framed as a transport input:
  `PeerPathReport { path: "direct" | "relay" | "unknown", rtt_ms }`, documented as
  "per-peer path state for split-relevant transport decisions".
- Bulk node-to-node payload already exists and is already ranged and verifiable:
  `PackageArtifactRequest { offset, .. }` with a per-artifact `sha256`.
- There is no `multipath` / link concept anywhere, and `config.toml` has no
  network section — the only defaults table is `[defaults.hardware]`.

Measured on the development host: Ethernet `192.168.1.5` (802.3, up, 1 Gbps,
interface metric 25) and a WLAN interface holding `192.168.1.6` with a default
route to the same gateway (metric 50). Two facts from the same reading matter for
this design:

- The **WLAN adapter reports `Disconnected`** while still holding an address and a
  route. Stale addresses outlive the association, so "has an IP" is not evidence
  that a link is usable, and any link inventory has to read the link state rather
  than the address list.
- Windows classifies the **Bluetooth PAN adapter as media type `802.3`** — that is,
  as wired. A platform media type is therefore wrong for exactly the interface
  kinds a phone or a cheap box will present (bridges, USB tethering, VPNs), which
  is why `kind` below is a policy hint and never a correctness input.

## The observation that shapes the design

Two payload classes cross node boundaries, and they want opposite things.

| payload | size | frequency | wants |
|---|---|---|---|
| activation frontier, stage to stage | one activation tensor per token (small) | every token | lowest, most stable latency |
| model package artifacts, node to node | GB | once per package or stage | maximum bandwidth, latency-insensitive |

Aggregating links therefore cannot be one global mode. Striping the frontier
would add reordering and jitter to the most latency-sensitive bytes in the
system; pinning bulk artifacts to one link would leave the other idle for the
only transfer big enough to care. Multi-link has to be **per payload class**.

## Design

### 1. Link

A `Link` is one local interface that can reach the mesh:

```
Link { id, kind: Wired | Wireless | Virtual | Unknown, local_addr, mtu,
       rtt_ms: Ewma, throughput_bps: Ewma, state: Up | Degraded | Down }
```

`kind` comes from the platform (Windows `Get-NetAdapter` media type, Linux
`/sys/class/net/<if>/wireless`, Android via the same sysfs) and is a **policy
hint only**. Nothing may depend on it for correctness, because the classification
is wrong on bridges, USB tethering and VPN interfaces, all of which are exactly
the links a phone or a cheap box will present.

### 2. One identity, several endpoints

Keep a single iroh `SecretKey` per node, so peers still see one `EndpointId`, and
run one iroh endpoint **bound to a single link's address** instead of `0.0.0.0`.
A node with two links then has two endpoints sharing one identity, and peers hold
several candidate paths to the same node.

This is the part that needs verification before anything is built — see the gate.

### 3. Path set per peer

Extend the existing report rather than inventing a parallel vocabulary:

```
PeerPathReport { node_id, paths: Vec<LinkPath { link_id, addr, kind,
                                                rtt_ms, state }> }
```

Existing consumers that read a single `path` / `rtt_ms` keep working by taking
the best entry, so the split planner needs no change on day one.

### 4. Scheduler, split by payload class

**Frontier — pin, never stripe.** Pick one path per peer by policy: prefer the
wired link when its RTT is within a small margin of the wireless one, otherwise
the lowest RTT; keep it sticky for the life of the session; switch only on loss
or a sustained RTT regression. Per-token traffic is small, so the win is jitter
avoidance, not bandwidth.

**Bulk artifacts — stripe.** Issue `N` concurrent ranged requests over the `N`
links, splitting by weight proportional to each link's measured throughput. The
existing `offset` field already expresses a range and the existing per-artifact
`sha256` already covers integrity, so this needs no new wire format. If a link
fails mid-transfer, reassign its outstanding ranges to the surviving links; the
range granularity is what bounds the redone work.

### 5. Measurement and config

Per-link throughput probe and EWMA RTT, surfaced in `/api/status` and
`network_diagnostics` next to today's `path` and `rtt_ms`, so the split planner's
existing `rtt_ms` consumer automatically starts seeing the best path.

New `[network]` section, defaulting to today's behaviour:

```toml
[network]
links = "auto"          # "auto" | explicit interface names
prefer = "wired"        # "wired" | "lowest-rtt" | "none"
aggregate = "bulk"      # "bulk" | "off"  -- only ever stripes bulk payloads
probe_interval_secs = 30
```

## Why not wait for QUIC multipath

MP-QUIC needs both peers and the whole stack to support it. Building the
scheduler above iroh gets the benefit now, on hardware that exists, and stays
compatible later: if iroh gains multipath, the scheduler degrades into a policy
input for it rather than being thrown away.

## The gate has been run — outcome and what it does and does not settle

Measured 2026-10-04 against the ARM node (X88 Pro 13 / RK3528, Android 13), which
was reachable over **two independent paths at once**: the direct cable
(`169.254.231.129`, host `169.254.231.128`) and the router
(`192.168.1.8`, host `192.168.1.6`).

```
wired only       120 MB in  17.73 s   6.8 MB/s
wireless only    120 MB in 188.73 s   0.6 MB/s
both (striped)   120 MB in 108.04 s   1.1 MB/s   (60 MB per link)
```

A parallel-stream probe was needed to interpret the wireless number, because
0.6 MB/s against 32 ms RTT looks like a per-stream window limit and a window limit
would mean the link has headroom a single stream cannot reach:

```
1 stream    30 MB in  29.7 s   1.0 MB/s
4 streams  120 MB in 108.1 s   1.1 MB/s aggregate, 0.3 MB/s per stream
```

Aggregate does not scale with stream count, so it is not a window limit: the
wireless path to this peer tops out near 1 MB/s (≈5–8 Mbps) against a 144.4 Mbps
negotiated rate.

**Verdict: do not build the aggregation half.** The reason is not that the paths
converge — they are genuinely separate, and both were live simultaneously. The
reason is a 7–11x rate asymmetry: striping bounds the wall clock by the slower
link, and the result is measurably worse than the cable alone (1.1 vs 6.8 MB/s).
No scheduler can recover that.

**Keep the selection half.** The same run measured the wired path at 0–1 ms RTT
with no loss and the wireless path at min 2 / max 104 / avg 32 ms. That is a
direct argument for pinning the activation frontier — the most latency-sensitive
bytes in the system — to the wired link when one exists. That part of the design
is supported by measurement and is worth implementing.

### Two device behaviours that invalidated earlier attempts

Both had to be fixed before any number here meant anything, and both will recur on
any Android node:

- **The box sleeps.** It suspended partway through the first measurement run and
  took both adb transports offline with it. Fixed with
  `svc power stayon true` plus `stay_on_while_plugged_in=7`.
- **Its Wi-Fi power save was on**, which is what produced the original 370 ms RTT
  spikes. `iw dev wlan0 set power_save off` cut the maximum to 104 ms. Android TV
  boxes ship with the radio dozing by default.

Also worth keeping: the box's `eth0` comes up with an IPv6 link-local and no IPv4,
so the cable carries nothing until an address is added, and Android installs no
packet-filter accept rule for an interface it does not manage, so inbound TCP on
that interface is dropped until one is added. ICMP works either way, which is why
the link looked alive while `adb connect` timed out.

### A measurement-discipline note

The first run of this gate printed `VERDICT: no material gain` from two failed
cases — one delivering 0 bytes in 0.12 s and one truncating at 110 of 200 MB. A
zero-byte case divides out to 0 MB/s and looks like a very slow link rather than a
refused connection. The harness now captures stderr and invalidates the whole run
when a case does not deliver the requested volume, and prints no ratio at all in
that case. Any future gate here should keep both properties.

### How the gate must be run

**The obvious local test does not work, and this is worth stating so nobody runs
it.** Two processes on one host, each bound to a different local address, do not
put traffic on the wire: the stack short-circuits host-to-self connections
through loopback regardless of which interface the address belongs to. A
same-host test would report the loopback rate for every configuration and
"confirm" the design no matter what the links do. The gate needs a second machine
and the transfer has to cross the wire.

The three cases are single-link wired, single-link wireless, and both striped.

Two diagnostics from getting the peer onto the wire at all are worth keeping,
because both cost time here and both will recur:

- Compare `Get-NetAdapterStatistics` deltas over a sample window before concluding
  anything from the address list: a stale link-local address plus a live port looks
  exactly like a working link, and only the counters tell them apart.
- A ping sweep is the wrong discovery tool for Android. Many Android devices drop
  ICMP by default, so a sweep can miss a peer that is present and reachable. mDNS
  is the reliable signal, and `adb mdns services` is the ready-made way to read it.
  The first sweep here concluded "the box is not on this segment" and was wrong.

## Risks and open questions

- **Two endpoints, one identity.** Must confirm iroh allows two endpoints with the
  same `EndpointId` on one host, and that a peer treats them as one node rather
  than two. If it does not, the fallback is several QUIC connections over one
  endpoint, which loses per-interface binding and with it most of the benefit.
- **Binding is not necessarily routing.** On Windows the weak host model usually
  honours a bound source address, but a same-subnet dual-homed host can still
  send both flows out one interface. The gate above measures exactly this.
- **Power.** On a phone or a battery box, keeping Wi-Fi busy for bulk transfer
  costs energy. The `aggregate = "bulk"` scope limits this to transfers that
  already spend far more energy on the radio being awake at all.
- **Link flapping.** Wireless links come and go. The scheduler must treat a link
  disappearing as routine, not as an error, and the frontier path must survive it
  without dropping a token.
- **This does not change placement.** Per the measured stage footprint, the
  per-step cost is resident weights plus streamed experts; the network carries
  the activation frontier and, once, the package. Multi-link improves the second
  and stabilises the first. It is not a substitute for layer staging, and it does
  not make expert-level distribution any more attractive.
