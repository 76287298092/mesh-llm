use super::*;

pub(crate) use mesh_llm_membership::peer_state::{
    ClaimedLogHead, DEAD_PEER_TTL, PEER_DOWN_REPORTER_COOLDOWN_SECS, PEER_STALE_SECS,
    ingest_tunnel_map, model_identity_score, policy_accepts_peer, resolve_peer_leaving,
    stream_allowed_before_admission,
};
pub use mesh_llm_membership::peer_state::{
    DirectLatencyObservation, DisplayLatency, DisplayLatencySource, MeshCatalogEntry,
    PeerAnnouncement, PeerInfo, PropagatedLatencyObservation,
};

pub(crate) fn peer_info_to_mesh_peer(peer: &PeerInfo) -> crate::plugin::proto::MeshPeer {
    crate::plugin::proto::MeshPeer {
        peer_id: endpoint_id_hex(peer.id),
        version: peer.version.clone().unwrap_or_default(),
        capabilities: Vec::new(),
        role: node_role_label(&peer.role),
        vram_bytes: peer.vram_bytes,
        models: peer.models.clone(),
        serving_models: peer.serving_models.clone(),
        available_models: Vec::new(),
        requested_models: peer.requested_models.clone(),
        rtt_ms: peer.current_direct_rtt_ms(),
        model_source: peer.model_source.clone().unwrap_or_default(),
        hosted_models: peer.hosted_models.clone(),
        hosted_models_known: Some(peer.hosted_models_known),
    }
}

pub(crate) fn load_or_refresh_owner_attestation(
    owner_keypair: &crate::crypto::OwnerKeypair,
    endpoint_id: EndpointId,
    node_label: Option<String>,
    hostname_hint: Option<String>,
) -> Result<SignedNodeOwnership> {
    // Always sign a fresh attestation on startup when the owner key is available.
    // This ensures that key rotation is always reflected immediately and no stale
    // certificate can persist across restarts.
    let path = default_node_ownership_path()?;
    let ownership = sign_node_ownership(
        owner_keypair,
        endpoint_id.as_bytes(),
        current_time_unix_ms() + DEFAULT_NODE_CERT_LIFETIME_SECS * 1000,
        node_label,
        hostname_hint,
    )?;
    save_node_ownership(&path, &ownership)?;
    Ok(ownership)
}

pub(crate) fn model_descriptor_score(descriptor: &ServedModelDescriptor) -> u8 {
    let identity = &descriptor.identity;
    let capability_bonus = u8::from(descriptor.capabilities.multimodal)
        + u8::from(descriptor.capabilities.audio != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.vision != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.reasoning != crate::models::CapabilityLevel::None)
        + u8::from(descriptor.capabilities.tool_use != crate::models::CapabilityLevel::None);
    let metadata_bonus = u8::from(descriptor.metadata.is_some());
    model_identity_score(identity) + capability_bonus + metadata_bonus
}

pub(crate) fn upsert_mesh_catalog_descriptor(
    descriptors: &mut HashMap<String, ServedModelDescriptor>,
    descriptor: ServedModelDescriptor,
) {
    if descriptor.identity.model_name.is_empty() {
        return;
    }
    let mut keys = vec![descriptor.identity.model_name.clone()];
    if let Some(public_id) = public_model_id_from_identity(&descriptor.identity) {
        keys.push(public_id);
    }
    keys.sort();
    keys.dedup();
    for key in keys {
        match descriptors.get(&key) {
            Some(existing)
                if model_descriptor_score(existing) >= model_descriptor_score(&descriptor) => {}
            _ => {
                descriptors.insert(key, descriptor.clone());
            }
        }
    }
}

/// Role a node plays in the mesh (moved to `mesh-llm-membership`).
pub use mesh_llm_membership::NodeRole;

#[derive(Debug)]
pub struct OwnerRuntimeConfig {
    pub keypair: Option<crate::crypto::OwnerKeypair>,
    pub control_bind: Option<std::net::SocketAddr>,
    pub control_advertise_addr: Option<std::net::SocketAddr>,
    pub node_label: Option<String>,
    pub trust_store: TrustStore,
    pub trust_policy: TrustPolicy,
    pub activity: mesh_llm_config::RuntimeActivityConfig,
    pub drain_timeout_secs: u64,
    pub drain_timeout_max_secs: u64,
    pub public_mesh: bool,
}

pub(crate) struct ControlListenerLifecycle {
    pub(crate) endpoint: Endpoint,
    pub(crate) token: String,
    pub(crate) shutdown_requested: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) shutdown: Arc<tokio::sync::Notify>,
    pub(crate) task: tokio::task::JoinHandle<()>,
}

/// Host-owned serving-routing projections over the neutral membership
/// `PeerInfo`.
///
/// Resolving a peer's advertised raw model ids to the mesh's public model ids
/// needs `skippy_model_ref` and the host model catalog, so these stay in the
/// host as free functions over `&PeerInfo` instead of methods on the
/// membership type.
pub(crate) fn routable_models(peer: &PeerInfo) -> Vec<String> {
    let raw = if peer.hosted_models_known {
        &peer.hosted_models
    } else {
        &peer.serving_models
    };
    let mut models = raw
        .iter()
        .map(|model| public_model_id_for_routable_model(peer, model))
        .collect::<Vec<_>>();
    models.sort();
    models.dedup();
    models
}

pub(crate) fn routes_model(peer: &PeerInfo, model: &str) -> bool {
    let raw = if peer.hosted_models_known {
        &peer.hosted_models
    } else {
        &peer.serving_models
    };
    raw.iter().any(|candidate| {
        candidate == model || public_model_id_for_routable_model(peer, candidate) == model
    })
}

pub(crate) fn http_routable_models(peer: &PeerInfo) -> Vec<String> {
    if peer.accepts_http_inference() {
        routable_models(peer)
    } else {
        Vec::new()
    }
}

pub(crate) fn routes_http_model(peer: &PeerInfo, model: &str) -> bool {
    peer.accepts_http_inference() && routes_model(peer, model)
}

fn public_model_id_for_routable_model(peer: &PeerInfo, model: &str) -> String {
    peer.served_model_descriptors
        .iter()
        .find(|descriptor| descriptor.identity.model_name == model)
        .and_then(|descriptor| public_model_id_from_identity(&descriptor.identity))
        .unwrap_or_else(|| canonical_demand_model_ref(model))
}

pub(crate) fn advertised_context_length(peer: &PeerInfo, model: &str) -> Option<u32> {
    advertised_context_length_for_runtime_model(peer, model).or_else(|| {
        peer.served_model_descriptors
            .iter()
            .filter(|descriptor| {
                let runtime_name = descriptor.identity.model_name.as_str();
                runtime_name != model
                    && public_model_id_for_routable_model(peer, runtime_name) == model
            })
            .find_map(|descriptor| {
                advertised_context_length_for_runtime_model(peer, &descriptor.identity.model_name)
            })
    })
}

fn advertised_context_length_for_runtime_model(peer: &PeerInfo, model: &str) -> Option<u32> {
    peer.served_model_runtime
        .iter()
        .find(|runtime| runtime.model_name == model)
        .and_then(ModelRuntimeDescriptor::advertised_context_length)
}

/// How long a confirmed-departed peer id stays barred from transitive
/// re-admission. [`DEAD_PEER_TTL`] expires quickly so reconnection attempts
/// can resume, but gossip bridges can keep carrying the departed id's final
/// announcement long after that (issue #1756): re-admitting it transitively
/// resurrects a ghost `state: serving` entry with no direct connection.
/// Only direct proof of life (a gossip exchange or connection with the id
/// itself) clears this record early; otherwise it expires silently.
pub(crate) const DEPARTED_PEER_TRANSITIVE_BLOCK_TTL: std::time::Duration =
    std::time::Duration::from_secs(3600); // 1 hour

pub(crate) struct MeshState {
    pub(crate) peers: HashMap<EndpointId, PeerInfo>,
    pub(crate) connections: HashMap<EndpointId, Connection>,
    pub(crate) pending_connections: HashMap<EndpointId, PendingConnectionHandshake>,
    pub(crate) next_pending_connection_attempt: u64,
    /// Remote peers' tunnel maps: peer_endpoint_id → { target_endpoint_id → tunnel_port_on_that_peer }
    pub(crate) remote_tunnel_maps: HashMap<EndpointId, HashMap<EndpointId, u16>>,
    /// Peers confirmed dead — don't reconnect from gossip discovery.
    /// Cleared when the peer successfully reconnects via rejoin/join.
    /// Entries expire after [`DEAD_PEER_TTL`] so that reconnection attempts
    /// resume. Transitive re-admission of the id stays blocked for
    /// [`DEPARTED_PEER_TRANSITIVE_BLOCK_TTL`] via [`MeshState::departed_peers`]
    /// so stale bridge announcements cannot resurrect it (issue #1756).
    pub(crate) dead_peers: HashMap<EndpointId, std::time::Instant>,
    /// Peer ids whose departure was confirmed (heartbeat failure or accepted
    /// PeerDown), with the instant of confirmation. Direct proof of life
    /// clears this wherever [`MeshState::dead_peers`] is cleared; otherwise
    /// entries expire after [`DEPARTED_PEER_TRANSITIVE_BLOCK_TTL`].
    pub(crate) departed_peers: HashMap<EndpointId, std::time::Instant>,
    /// Tracks (reporter, target) pairs where a PeerDown claim was rejected
    /// (target was still reachable). Used to suppress repeated false reports
    /// from unreliable reporters (e.g. relay-partitioned nodes).
    pub(crate) peer_down_rejections: HashMap<(EndpointId, EndpointId), std::time::Instant>,
    /// Last accepted direct-path dial-back request per peer. This keeps path
    /// maintenance targeted even if a peer repeatedly asks us to reverse-dial.
    pub(crate) direct_path_request_last_at: HashMap<EndpointId, std::time::Instant>,
    pub(crate) seen_plugin_messages: HashMap<String, std::time::Instant>,
    pub(crate) seen_plugin_message_order: VecDeque<(std::time::Instant, String)>,
    /// Last policy-rejection status per peer — used to suppress duplicate log lines.
    /// Only logs when the status transitions (first rejection or status change).
    pub(crate) policy_rejected_peers: HashMap<EndpointId, OwnershipStatus>,
    /// Peers rejected by immutable mesh requirements. Used to keep pre-admission
    /// streams from disclosing topology after a deterministic requirement reject.
    pub(crate) requirement_rejected_peers: HashSet<EndpointId>,
    pub(crate) recent_mesh_rejections: VecDeque<MeshRequirementRejectionEvent>,
}

impl MeshState {
    /// The initial empty mesh state. `next_pending_connection_attempt` starts
    /// at 1 so a zero attempt counter can mean "unset".
    pub(crate) fn new() -> Self {
        Self {
            peers: HashMap::new(),
            connections: HashMap::new(),
            pending_connections: HashMap::new(),
            next_pending_connection_attempt: 1,
            remote_tunnel_maps: HashMap::new(),
            dead_peers: HashMap::new(),
            departed_peers: HashMap::new(),
            peer_down_rejections: HashMap::new(),
            direct_path_request_last_at: HashMap::new(),
            seen_plugin_messages: HashMap::new(),
            seen_plugin_message_order: VecDeque::new(),
            policy_rejected_peers: HashMap::new(),
            requirement_rejected_peers: HashSet::new(),
            recent_mesh_rejections: VecDeque::new(),
        }
    }
}

/// Returns `true` if the given peer has completed gossip validation and is
/// a full mesh member. Inbound unadmitted peers may be in `state.connections`
/// before `state.peers` — they are quarantined until gossip succeeds.
#[cfg(test)]
pub(crate) fn is_peer_admitted(peers: &HashMap<EndpointId, PeerInfo>, id: &EndpointId) -> bool {
    peers.get(id).is_some_and(PeerInfo::is_admitted)
}

impl Node {
    pub async fn mesh_catalog(&self) -> Vec<String> {
        // Snapshot each lock independently to avoid holding multiple locks.
        let my_available = self.available_models.lock().await.clone();
        let my_requested = self.requested_models.lock().await.clone();
        let my_serving_models = self.serving_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| {
                    (
                        p.available_models.clone(),
                        p.requested_models.clone(),
                        p.serving_models.clone(),
                    )
                })
                .collect()
        };
        let mut all = std::collections::HashSet::new();
        for m in &my_available {
            all.insert(m.clone());
        }
        for m in &my_requested {
            all.insert(m.clone());
        }
        for m in &my_serving_models {
            all.insert(m.clone());
        }
        for (avail, req, serving_models) in &peer_data {
            for m in avail {
                all.insert(m.clone());
            }
            for m in req {
                all.insert(m.clone());
            }
            for m in serving_models {
                all.insert(m.clone());
            }
        }
        let mut result: Vec<String> = all.into_iter().collect();
        result.sort();
        result
    }

    pub async fn mesh_catalog_entries(&self) -> Vec<MeshCatalogEntry> {
        let names = self.mesh_catalog().await;
        let my_available = self.available_models.lock().await.clone();
        let my_served_descriptors = self.served_model_descriptors.lock().await.clone();
        let peer_descriptors: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .map(|p| p.served_model_descriptors.clone())
                .collect()
        };

        let mut by_name: HashMap<String, ServedModelDescriptor> = HashMap::new();
        for descriptor in infer_available_model_descriptors(&my_available)
            .into_iter()
            .chain(my_served_descriptors)
        {
            upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
        }
        for served in peer_descriptors {
            for descriptor in served {
                upsert_mesh_catalog_descriptor(&mut by_name, descriptor);
            }
        }

        names
            .into_iter()
            .map(|model_name| MeshCatalogEntry {
                descriptor: by_name.get(&model_name).cloned(),
                model_name,
            })
            .collect()
    }

    /// Get all models currently reachable via the mesh HTTP/API ingress.
    ///
    /// This is intentionally stricter than "loaded in VRAM somewhere": split
    /// workers may contribute compute for a model but cannot accept chat
    /// requests directly.
    pub async fn models_being_served(&self) -> Vec<String> {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state.peers.values().cloned().collect()
        };
        let mut served = std::collections::HashSet::new();
        for s in &my_hosted_models {
            served.insert(s.clone());
        }
        for peer in &peer_data {
            for m in http_routable_models(peer) {
                served.insert(m.clone());
            }
        }
        let mut result: Vec<String> = served.into_iter().collect();
        result.sort();
        result
    }

    /// All host IDs serving a model, with a healthy hash-preferred host first.
    /// Paused peers are excluded; deprioritized peers remain spillover capacity.
    pub async fn hosts_for_model(&self, model: &str) -> Vec<EndpointId> {
        let state = self.state.lock().await;
        let mut hosts: Vec<(EndpointId, bool)> = state
            .peers
            .values()
            .filter(|p| p.is_admitted())
            .filter(|p| routes_http_model(p, model))
            .filter_map(|p| {
                use crate::proto::node::InferenceAdmissionState;
                match p.inference_admission_state {
                    Some(
                        InferenceAdmissionState::RemotePaused | InferenceAdmissionState::AllPaused,
                    ) => None,
                    Some(InferenceAdmissionState::AcceptingDeprioritized) => Some((p.id, true)),
                    _ => Some((p.id, false)),
                }
            })
            .collect();
        hosts.sort_by_key(|(id, _)| *id);
        let mut healthy = Vec::new();
        let mut deprioritized = Vec::new();
        for (id, is_deprioritized) in hosts {
            if is_deprioritized {
                deprioritized.push(id);
            } else {
                healthy.push(id);
            }
        }
        // Rendezvous-order each health class independently. This keeps one
        // origin sticky to one replica for KV reuse, spreads different origins,
        // and when a replica becomes busy removes only that replica from the
        // healthy ordering instead of rotating every remaining peer.
        let origin = self.endpoint.id();
        let affinity_score = |peer: &EndpointId| {
            origin
                .as_bytes()
                .iter()
                .chain(peer.as_bytes())
                .fold(0xcbf29ce484222325u64, |hash, &byte| {
                    hash.wrapping_mul(0x100000001b3) ^ u64::from(byte)
                })
        };
        healthy.sort_by_key(|peer| std::cmp::Reverse(affinity_score(peer)));
        deprioritized.sort_by_key(|peer| std::cmp::Reverse(affinity_score(peer)));
        healthy.extend(deprioritized);
        healthy
    }

    /// Find ANY host in the mesh (fallback when no model match).
    pub async fn any_host(&self) -> Option<PeerInfo> {
        let state = self.state.lock().await;
        state
            .peers
            .values()
            .filter(|p| p.is_admitted())
            .find(|p| !http_routable_models(p).is_empty())
            .cloned()
    }

    /// Build the current routing table from this node's view of the mesh.
    pub async fn routing_table(&self) -> RoutingTable {
        let my_hosted_models = self.hosted_models.lock().await.clone();
        let my_role = self.role.lock().await.clone();
        let peer_data: Vec<_> = {
            let state = self.state.lock().await;
            state
                .peers
                .values()
                .filter(|peer| peer.is_admitted())
                .cloned()
                .collect()
        };
        let mut hosts = Vec::new();

        // Include self if we're serving through the local API proxy
        if !matches!(my_role, NodeRole::Client) {
            for model in my_hosted_models {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", self.endpoint.id().fmt_short()),
                    endpoint_id: self.endpoint.id(),
                    vram_gb: self.vram_bytes as f64 / 1e9,
                });
            }
        }

        // Include peers that are serving through their local API proxies
        for peer in &peer_data {
            for model in http_routable_models(peer) {
                hosts.push(RouteEntry {
                    model,
                    node_id: format!("{}", peer.id.fmt_short()),
                    endpoint_id: peer.id,
                    vram_gb: peer.vram_bytes as f64 / 1e9,
                });
            }
        }

        let mesh_id = self.mesh_id().await;
        RoutingTable { hosts, mesh_id }
    }

    /// Accelerator-resident capacity used for mesh stage placement.
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }

    /// Local model-fit budget, including supported CPU offload memory.
    pub fn local_runtime_capacity_bytes(&self) -> u64 {
        self.local_runtime_capacity_bytes
    }

    pub async fn peers(&self) -> Vec<PeerInfo> {
        self.state
            .lock()
            .await
            .peers
            .values()
            .filter(|peer| peer.is_admitted())
            .cloned()
            .collect()
    }

    pub(crate) async fn connection_to_peer(&self, peer_id: EndpointId) -> Result<Connection> {
        let state = self.state.lock().await;
        match state.connections.get(&peer_id).cloned() {
            Some(conn) if state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted) => Ok(conn),
            Some(conn) => {
                drop(state);
                if let Err(error) = self
                    .complete_cached_connection_gossip_single_flight(peer_id, conn, false)
                    .await
                {
                    anyhow::bail!(
                        "Failed to complete gossip with {} before opening mesh stream: {error}",
                        peer_id.fmt_short()
                    );
                }
                let state = self.state.lock().await;
                state
                    .connections
                    .get(&peer_id)
                    .cloned()
                    .filter(|_| state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted))
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "No admitted connection for {} after gossip",
                            peer_id.fmt_short()
                        )
                    })
            }
            None => {
                let addr = state.peers.get(&peer_id).map(|p| p.addr.clone());
                drop(state);
                let Some(addr) = addr else {
                    anyhow::bail!("No connection or address for {}", peer_id.fmt_short());
                };
                let owner = match self.reserve_pending_connection(peer_id).await {
                    PendingConnectionReservation::Owner(owner) => owner,
                    PendingConnectionReservation::Waiter(waiter) => {
                        self.await_pending_connection(waiter).await?;
                        let state = self.state.lock().await;
                        return state
                            .connections
                            .get(&peer_id)
                            .cloned()
                            .filter(|_| {
                                state.peers.get(&peer_id).is_some_and(PeerInfo::is_admitted)
                            })
                            .ok_or_else(|| {
                                anyhow::anyhow!(
                                    "No admitted connection for {} after pending handshake",
                                    peer_id.fmt_short()
                                )
                            });
                    }
                };
                let conn = match tokio::time::timeout(
                    std::time::Duration::from_secs(10),
                    connect_mesh(&self.endpoint, addr),
                )
                .await
                {
                    Ok(Ok(conn)) => conn,
                    Ok(Err(error)) => {
                        let error = anyhow::anyhow!(
                            "Failed to connect to {}: {error}",
                            peer_id.fmt_short()
                        );
                        self.finish_pending_connection(
                            owner,
                            PendingConnectionOutcome::Failed(error.to_string()),
                        )
                        .await;
                        return Err(error);
                    }
                    Err(_) => {
                        let error =
                            anyhow::anyhow!("Timeout connecting to {}", peer_id.fmt_short());
                        self.finish_pending_connection(
                            owner,
                            PendingConnectionOutcome::Failed(error.to_string()),
                        )
                        .await;
                        return Err(error);
                    }
                };
                if let Err(error) = self
                    .initiate_gossip_inner(conn.clone(), peer_id, false)
                    .await
                {
                    conn.close(0u32.into(), b"on-demand-gossip-failed");
                    self.finish_pending_connection(
                        owner,
                        PendingConnectionOutcome::Failed(error.to_string()),
                    )
                    .await;
                    anyhow::bail!(
                        "Failed to complete gossip with {} before opening mesh stream: {error}",
                        peer_id.fmt_short()
                    );
                }
                self.state
                    .lock()
                    .await
                    .connections
                    .insert(peer_id, conn.clone());
                let node_for_dispatch = self.clone();
                let conn_for_dispatch = conn.clone();
                tokio::spawn(async move {
                    node_for_dispatch
                        .dispatch_streams(conn_for_dispatch, peer_id)
                        .await;
                });
                self.finish_pending_connection(owner, PendingConnectionOutcome::Admitted)
                    .await;
                Ok(conn)
            }
        }
    }

    pub(crate) async fn split_stage_path_snapshot(
        &self,
        peer_id: EndpointId,
    ) -> SplitStagePathSnapshot {
        let fallback = self.peer_stage_path_fallback(peer_id).await;
        match self.stage_connection_to_peer(peer_id).await {
            Ok(conn) => {
                split_stage_path_snapshot_from_connection(&conn).with_peer_path_fallback(fallback)
            }
            Err(error) => {
                tracing::debug!(
                    peer = %peer_id.fmt_short(),
                    error = %error,
                    "split stage path probe could not open stage connection"
                );
                SplitStagePathSnapshot::unknown().with_peer_path_fallback(fallback)
            }
        }
    }

    pub(crate) async fn peer_stage_path_fallback(
        &self,
        peer_id: EndpointId,
    ) -> Option<SelectedPathObservation> {
        let state = self.state.lock().await;
        state
            .peers
            .get(&peer_id)
            .and_then(PeerInfo::split_stage_path_fallback)
    }

    pub(crate) async fn open_mesh_subprotocol_stream(
        &self,
        peer_id: EndpointId,
        name: &str,
        major: u32,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        use prost::Message as _;

        let conn = self.connection_to_peer(peer_id).await?;
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_SUBPROTOCOL]).await?;
        let open = crate::proto::node::MeshSubprotocolOpen {
            r#gen: NODE_PROTOCOL_GENERATION,
            name: name.to_string(),
            major,
        };
        open.validate_frame()
            .map_err(|error| anyhow::anyhow!("invalid mesh subprotocol open: {error}"))?;
        write_len_prefixed(&mut send, &open.encode_to_vec()).await?;
        Ok((send, recv))
    }

    pub(crate) async fn open_skippy_stage_mesh_stream(
        &self,
        peer_id: EndpointId,
        stream_kind: u8,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        let (mut send, recv) = self
            .open_mesh_subprotocol_stream(
                peer_id,
                skippy_protocol::STAGE_SUBPROTOCOL_NAME,
                skippy_protocol::STAGE_SUBPROTOCOL_MAJOR,
            )
            .await?;
        send.write_all(&[stream_kind]).await?;
        Ok((send, recv))
    }

    pub(crate) async fn stage_connection_to_peer(&self, peer_id: EndpointId) -> Result<Connection> {
        let addr = {
            let state = self.state.lock().await;
            state.peers.get(&peer_id).map(|p| p.addr.clone())
        };
        let Some(addr) = addr else {
            anyhow::bail!("No address for stage peer {}", peer_id.fmt_short());
        };
        let conn = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            self.endpoint
                .connect(addr, skippy_protocol::STAGE_ALPN_V2)
                .await
        })
        .await
        .map_err(|_| anyhow::anyhow!("Timeout connecting to stage peer {}", peer_id.fmt_short()))?
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to connect to stage peer {}: {e}",
                peer_id.fmt_short()
            )
        })?;
        Ok(conn)
    }
}
impl Node {
    pub(crate) async fn handle_peer_down_stream(
        &self,
        remote: EndpointId,
        mut recv: iroh::endpoint::RecvStream,
    ) {
        let Some(dead_id) = self.decode_peer_down_frame(&mut recv).await else {
            return;
        };
        let report = self.peer_down_report(remote, dead_id).await;
        self.apply_peer_down_report(remote, dead_id, report).await;
    }

    pub(crate) async fn decode_peer_down_frame(
        &self,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<EndpointId> {
        let frame = self.read_peer_down_frame(recv).await?;
        peer_down_endpoint_id(&frame)
    }

    pub(crate) async fn read_peer_down_frame(
        &self,
        recv: &mut iroh::endpoint::RecvStream,
    ) -> Option<crate::proto::node::PeerDown> {
        let proto_buf = match read_len_prefixed(recv).await {
            Ok(buf) => buf,
            Err(e) => {
                tracing::warn!("PeerDown: failed to read proto body — rejecting: {e}");
                return None;
            }
        };
        self.decode_peer_down_proto(&proto_buf)
    }

    pub(crate) fn decode_peer_down_proto(
        &self,
        proto_buf: &[u8],
    ) -> Option<crate::proto::node::PeerDown> {
        let frame = match crate::proto::node::PeerDown::decode(proto_buf) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("PeerDown: invalid protobuf — rejecting: {e}");
                return None;
            }
        };
        if let Err(e) = frame.validate_frame() {
            tracing::warn!("PeerDown: frame validation failed — rejecting: {e}");
            return None;
        }
        Some(frame)
    }

    pub(crate) async fn peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
    ) -> PeerDownReport {
        let state = self.state.lock().await;
        let conn_opt = state.connections.get(&dead_id).cloned();
        let peer = state.peers.get(&dead_id);
        let peer_addr = peer.map(|p| p.addr.clone());
        let recently_seen = peer
            .map(|p| p.last_seen.elapsed().as_secs() < PEER_STALE_SECS)
            .unwrap_or(false);
        let reporter_cooled = state
            .peer_down_rejections
            .get(&(remote, dead_id))
            .is_some_and(|t| t.elapsed().as_secs() < PEER_DOWN_REPORTER_COOLDOWN_SECS);
        PeerDownReport {
            conn_opt,
            peer_addr,
            recently_seen,
            reporter_cooled,
        }
    }

    pub(crate) async fn apply_peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
        report: PeerDownReport,
    ) {
        match peer_down_report_disposition(report.reporter_cooled, report.recently_seen) {
            PeerDownReportDisposition::SuppressReporterCooldown => tracing::debug!(
                "PeerDown: {} reported {} dead but reporter is in cooldown, ignoring",
                remote.fmt_short(),
                dead_id.fmt_short()
            ),
            PeerDownReportDisposition::RejectRecentlySeen => {
                self.reject_recent_peer_down_report(remote, dead_id).await;
            }
            PeerDownReportDisposition::ProbeReachability => {
                self.probe_and_apply_peer_down(remote, dead_id, report)
                    .await;
            }
        }
    }

    pub(crate) async fn reject_recent_peer_down_report(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
    ) {
        emit_mesh_info(format!(
            "ℹ️  Peer {} reported dead by {} but seen recently (direct alive), ignoring",
            dead_id.fmt_short(),
            remote.fmt_short()
        ));
        self.record_peer_down_rejection(remote, dead_id).await;
    }

    pub(crate) async fn probe_and_apply_peer_down(
        &self,
        remote: EndpointId,
        dead_id: EndpointId,
        report: PeerDownReport,
    ) {
        let should_remove = self
            .peer_down_probe_should_remove(dead_id, report.conn_opt, report.peer_addr)
            .await;
        if let Some(id) = resolve_peer_down(self.endpoint.id(), dead_id, should_remove) {
            self.remove_confirmed_peer_down(remote, id).await;
        } else if dead_id != self.endpoint.id() {
            emit_mesh_info(format!(
                "ℹ️  Peer {} reported dead by {} but still reachable, ignoring",
                dead_id.fmt_short(),
                remote.fmt_short()
            ));
            self.record_peer_down_rejection(remote, dead_id).await;
        }
    }

    pub(crate) async fn peer_down_probe_should_remove(
        &self,
        dead_id: EndpointId,
        conn_opt: Option<Connection>,
        peer_addr: Option<EndpointAddr>,
    ) -> bool {
        if let Some(conn) = conn_opt {
            return !matches!(
                tokio::time::timeout(std::time::Duration::from_secs(5), conn.open_bi()).await,
                Ok(Ok(_))
            );
        }
        let Some(addr) = peer_addr else {
            return true;
        };
        match tokio::time::timeout(
            std::time::Duration::from_secs(8),
            connect_mesh(&self.endpoint, addr),
        )
        .await
        {
            Ok(Ok(new_conn)) => {
                self.keep_reachable_peer_down_connection(dead_id, new_conn)
                    .await;
                false
            }
            _ => true,
        }
    }

    pub(crate) async fn keep_reachable_peer_down_connection(
        &self,
        dead_id: EndpointId,
        new_conn: Connection,
    ) {
        emit_mesh_info(format!(
            "ℹ️  Peer {} reported dead but we reached them, keeping",
            dead_id.fmt_short()
        ));
        let mut state = self.state.lock().await;
        if state.connections.contains_key(&dead_id) {
            return;
        }
        state.connections.insert(dead_id, new_conn.clone());
        drop(state);
        let node = self.clone();
        tokio::spawn(async move {
            node.dispatch_streams(new_conn, dead_id).await;
        });
    }

    pub(crate) async fn remove_confirmed_peer_down(&self, remote: EndpointId, id: EndpointId) {
        emit_mesh_warning(format!(
            "⚠️  Peer {} reported dead by {}, confirmed, removing",
            id.fmt_short(),
            remote.fmt_short()
        ));
        let mut state = self.state.lock().await;
        state.dead_peers.insert(id, std::time::Instant::now());
        state.departed_peers.insert(id, std::time::Instant::now());
        state.connections.remove(&id);
        drop(state);
        self.remove_peer(id, MeshPeerRemovalReason::PeerDownProbeFailed)
            .await;
    }

    pub(crate) async fn record_peer_down_rejection(&self, remote: EndpointId, dead_id: EndpointId) {
        self.state
            .lock()
            .await
            .peer_down_rejections
            .insert((remote, dead_id), std::time::Instant::now());
    }
}
impl Node {
    pub(crate) async fn handle_tunnel_map_stream(
        &self,
        remote: EndpointId,
        protocol: ControlProtocol,
        mut recv: iroh::endpoint::RecvStream,
    ) -> Result<()> {
        use prost::Message as _;

        let buf = read_len_prefixed(&mut recv).await?;
        let _ = protocol;
        let frame = crate::proto::node::TunnelMap::decode(buf.as_slice())
            .map_err(|e| anyhow::anyhow!("TunnelMap decode error: {e}"))?;

        frame
            .validate_frame()
            .map_err(|e| anyhow::anyhow!("TunnelMap validation failed: {e}"))?;

        let entry_count = frame.entries.len();
        {
            let mut state = self.state.lock().await;
            ingest_tunnel_map(remote, &frame, &mut state.remote_tunnel_maps)?;
        }

        tracing::info!(
            "Received tunnel map from {} ({} entries)",
            remote.fmt_short(),
            entry_count
        );

        Ok(())
    }
}
