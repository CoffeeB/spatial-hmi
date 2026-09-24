/**
 * Spatial Node Manager handling interactive 3D nodes mounted on the globe.
 */

/**
 * Spatial Node Manager handling interactive 3D nodes mounted on the globe
 * and hierarchical expandable sub-node clusters.
 */

class SpatialNodeManager {
  constructor(globe) {
    this.globe = globe;
    this.nodes = [];
    this.subnodes = [];
    this.hoveredNode = null;
    this.selectedNode = null;
    this.activeSubnode = null;
    this.expandedCluster = null;

    this.nodesGroup = new THREE.Group();
    this.clusterGroup = new THREE.Group();
    this.globe.globeGroup.add(this.nodesGroup);
    this.globe.globeGroup.add(this.clusterGroup);

    this.raycaster = new THREE.Raycaster();
    this._initSampleNodes();
  }

  _initSampleNodes() {
    const nodeData = [
      {
        id: "node-alpha",
        label: "Alpha Station",
        lat: 37.77,
        lon: -122.41,
        color: 0x00f0ff,
        subnodes: [
          { id: "alpha-1", label: "Core Relay", metric: "99.8% Sync", offset: [-0.35, 0.25, 0.15] },
          { id: "alpha-2", label: "Telemetry Sensor", metric: "4.2 Gbps", offset: [0.35, 0.28, -0.1] },
          { id: "alpha-3", label: "Power Subsystem", metric: "98.4 kW", offset: [0.0, -0.38, 0.2] },
          { id: "alpha-4", label: "Optical Link", metric: "12.4 ms", offset: [-0.25, -0.25, -0.2] },
        ],
      },
      {
        id: "node-beta",
        label: "Beta Outpost",
        lat: 51.50,
        lon: -0.12,
        color: 0x3b82f6,
        subnodes: [
          { id: "beta-1", label: "Comm Transceiver", metric: "100% Signal", offset: [-0.3, 0.3, 0.1] },
          { id: "beta-2", label: "Radar Array", metric: "360° Scan", offset: [0.32, 0.2, -0.15] },
          { id: "beta-3", label: "Ion Battery", metric: "94% Charge", offset: [0.0, -0.35, 0.15] },
        ],
      },
      {
        id: "node-gamma",
        label: "Gamma Relay",
        lat: 35.67,
        lon: 139.65,
        color: 0x10b981,
        subnodes: [
          { id: "gamma-1", label: "Quantum Router", metric: "10.8 PFLOPS", offset: [-0.35, 0.2, 0.1] },
          { id: "gamma-2", label: "Atmosphere Monitor", metric: "0.04 ppm", offset: [0.35, -0.15, -0.1] },
          { id: "gamma-3", label: "Thermal Array", metric: "294.1 K", offset: [0.0, 0.35, -0.2] },
        ],
      },
      {
        id: "node-delta",
        label: "Delta Hub",
        lat: -33.86,
        lon: 151.20,
        color: 0xa855f7,
        subnodes: [
          { id: "delta-1", label: "Deep Space Link", metric: "Active Beam", offset: [-0.28, 0.32, 0.12] },
          { id: "delta-2", label: "Telemetry Matrix", metric: "60 Hz Stream", offset: [0.3, -0.25, -0.1] },
          { id: "delta-3", label: "Storage Core", metric: "4.8 PB / 5.0 PB", offset: [0.0, 0.35, 0.2] },
        ],
      },
      {
        id: "node-epsilon",
        label: "Epsilon Sensor",
        lat: 1.35,
        lon: 103.81,
        color: 0xf59e0b,
        subnodes: [
          { id: "eps-1", label: "Oceanic Probe", metric: "Depth 4200m", offset: [-0.3, -0.2, 0.2] },
          { id: "eps-2", label: "Solar Harvester", metric: "142 kW Output", offset: [0.3, 0.25, -0.1] },
        ],
      },
      {
        id: "node-zeta",
        label: "Zeta Terminal",
        lat: -22.90,
        lon: -43.17,
        color: 0xf43f5e,
        subnodes: [
          { id: "zeta-1", label: "Defense Array", metric: "Ready", offset: [-0.32, 0.25, 0.15] },
          { id: "zeta-2", label: "Uplink Beacon", metric: "Carrier Lock", offset: [0.32, -0.22, -0.12] },
        ],
      },
    ];

    // Only render nodes that are non-empty and have further importance (sub-nodes)
    nodeData
      .filter((data) => data.subnodes && data.subnodes.length > 0)
      .forEach((data) => {
        const node = this._createNodeMesh(data);
        this.nodes.push(node);
        this.nodesGroup.add(node.mesh);
      });
  }

  _latLonToVector3(lat, lon, radius) {
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);

    const x = -(radius * Math.sin(phi) * Math.cos(theta));
    const z = radius * Math.sin(phi) * Math.sin(theta);
    const y = radius * Math.cos(phi);

    return new THREE.Vector3(x, y, z);
  }

  _createNodeMesh(data) {
    const pos = this._latLonToVector3(data.lat, data.lon, this.globe.radius * 1.01);

    const group = new THREE.Group();
    group.position.copy(pos);

    // 1. Holographic Core Sphere
    const coreGeo = new THREE.SphereGeometry(0.075, 16, 16);
    const coreMat = new THREE.MeshBasicMaterial({
      color: data.color,
      transparent: true,
      opacity: 0.9,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    group.add(coreMesh);

    // 2. Outer Pulsing Beacon Ring
    const ringGeo = new THREE.RingGeometry(0.09, 0.13, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: data.color,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.lookAt(pos.clone().multiplyScalar(2));
    group.add(ringMesh);

    return {
      id: data.id,
      label: data.label,
      lat: data.lat,
      lon: data.lon,
      color: data.color,
      subnodes: data.subnodes || [],
      mesh: group,
      coreMesh: coreMesh,
      ringMesh: ringMesh,
      basePos: pos.clone(),
      isSelected: false,
      isHovered: false,
      pulseTime: Math.random() * Math.PI,
    };
  }

  expandCluster(node) {
    if (this.expandedCluster === node) return;
    this.collapseCluster();

    this.expandedCluster = node;
    this.subnodes = [];
    this.activeClusterLinks = [];

    const lineMat = new THREE.LineBasicMaterial({
      color: node.color || 0x00f0ff,
      transparent: true,
      opacity: 0.65,
    });

    node.subnodes.forEach((sub, idx) => {
      const subGroup = new THREE.Group();
      const parentWorldPos = node.basePos.clone();
      subGroup.position.copy(parentWorldPos);

      // Target expanded offset
      const targetPos = parentWorldPos.clone().add(
        new THREE.Vector3(...sub.offset).multiplyScalar(1.6)
      );

      // Sub-node Mesh
      const subCoreGeo = new THREE.SphereGeometry(0.045, 12, 12);
      const subCoreMat = new THREE.MeshBasicMaterial({
        color: node.color || 0x00f0ff,
        transparent: true,
        opacity: 0.95,
      });
      const subCore = new THREE.Mesh(subCoreGeo, subCoreMat);
      subGroup.add(subCore);

      // Connecting energy line from Parent Node (0,0,0) to Sub-Node
      const lineGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(...sub.offset).multiplyScalar(1.6),
      ]);
      const lineMesh = new THREE.Line(lineGeo, lineMat);
      node.mesh.add(lineMesh);
      this.activeClusterLinks.push(lineMesh);

      const subObj = {
        id: sub.id,
        label: sub.label,
        metric: sub.metric,
        parent: node,
        group: subGroup,
        coreMesh: subCore,
        lineMesh: lineMesh,
        targetPos: targetPos,
        currentPos: parentWorldPos.clone(),
        isSubnode: true,
        pulseTime: Math.random() * Math.PI,
      };

      this.subnodes.push(subObj);
      this.clusterGroup.add(subGroup);
    });

    // Links between adjacent active sub-nodes connecting through cluster
    for (let i = 0; i < node.subnodes.length; i++) {
      const nextIdx = (i + 1) % node.subnodes.length;
      const pA = new THREE.Vector3(...node.subnodes[i].offset).multiplyScalar(1.6);
      const pB = new THREE.Vector3(...node.subnodes[nextIdx].offset).multiplyScalar(1.6);

      const interLineGeo = new THREE.BufferGeometry().setFromPoints([pA, pB]);
      const interLineMat = new THREE.LineBasicMaterial({
        color: node.color || 0x00f0ff,
        transparent: true,
        opacity: 0.35,
      });
      const interLineMesh = new THREE.Line(interLineGeo, interLineMat);
      node.mesh.add(interLineMesh);
      this.activeClusterLinks.push(interLineMesh);
    }

    // Update HUD Cluster Card
    this._updateClusterHUD(node);
  }

  collapseCluster() {
    if (!this.expandedCluster) return;

    if (this.activeClusterLinks && this.expandedCluster.mesh) {
      this.activeClusterLinks.forEach((link) => {
        this.expandedCluster.mesh.remove(link);
      });
    }
    this.activeClusterLinks = [];

    this.subnodes.forEach((sub) => {
      this.clusterGroup.remove(sub.group);
    });

    this.subnodes = [];
    this.expandedCluster = null;
    this.activeSubnode = null;

    const clusterCard = document.getElementById("cluster-card");
    if (clusterCard) clusterCard.classList.add("hidden");
  }

  _updateClusterHUD(node) {
    const clusterCard = document.getElementById("cluster-card");
    const clusterTitle = document.getElementById("cluster-title");
    const clusterCount = document.getElementById("cluster-node-count");
    const subnodesList = document.getElementById("subnodes-list");

    if (!clusterCard) return;

    clusterTitle.textContent = node.label;
    clusterCount.textContent = `${node.subnodes.length} SUB-NODES`;
    subnodesList.innerHTML = "";

    node.subnodes.forEach((sub) => {
      const item = document.createElement("div");
      item.className = "subnode-item";
      item.id = `item-${sub.id}`;
      item.innerHTML = `
        <div class="subnode-name-group">
          <div class="subnode-dot"></div>
          <span class="subnode-name">${sub.label}</span>
        </div>
        <span class="subnode-metric">${sub.metric}</span>
      `;
      item.addEventListener("click", () => {
        this.selectSubnode(this.subnodes.find((s) => s.id === sub.id));
      });
      subnodesList.appendChild(item);
    });

    clusterCard.classList.remove("hidden");
  }

  testRaycast(camera, ndcCoords) {
    this.raycaster.setFromCamera(ndcCoords, camera);

    // Test sub-nodes first if cluster expanded
    if (this.subnodes.length > 0) {
      const subMeshes = this.subnodes.map((s) => s.coreMesh);
      const subHits = this.raycaster.intersectObjects(subMeshes, true);
      if (subHits.length > 0) {
        const hitSubMesh = subHits[0].object;
        const hoveredSub = this.subnodes.find((s) => s.coreMesh === hitSubMesh);
        if (hoveredSub) {
          this.selectSubnode(hoveredSub);
          return hoveredSub;
        }
      }
    }

    // Test parent nodes
    const meshes = this.nodes.map((n) => n.coreMesh);
    const intersects = this.raycaster.intersectObjects(meshes, true);

    let newHover = null;
    if (intersects.length > 0) {
      const hitMesh = intersects[0].object;
      newHover = this.nodes.find((n) => n.coreMesh === hitMesh);
    }

    if (newHover !== this.hoveredNode) {
      if (this.hoveredNode && !this.hoveredNode.isSelected) {
        this.hoveredNode.coreMesh.scale.set(1, 1, 1);
      }
      this.hoveredNode = newHover;
      if (this.hoveredNode) {
        this.hoveredNode.coreMesh.scale.set(1.5, 1.5, 1.5);
      }
    }

    return this.hoveredNode;
  }

  selectNode(node) {
    if (!node) return;

    if (this.selectedNode) {
      this.selectedNode.isSelected = false;
      this.selectedNode.coreMesh.material.color.setHex(this.selectedNode.color);
    }

    this.selectedNode = node;
    this.selectedNode.isSelected = true;
    this.selectedNode.coreMesh.material.color.setHex(0xffffff);
    this.selectedNode.coreMesh.scale.set(1.8, 1.8, 1.8);

    // Automatically expand hierarchical sub-node cluster
    this.expandCluster(node);
  }

  selectSubnode(subnode) {
    if (!subnode) return;
    this.activeSubnode = subnode;

    this.subnodes.forEach((s) => {
      s.coreMesh.material.color.setHex(0x00f0ff);
      s.coreMesh.scale.set(1, 1, 1);
      const el = document.getElementById(`item-${s.id}`);
      if (el) el.classList.remove("selected");
    });

    subnode.coreMesh.material.color.setHex(0xffffff);
    subnode.coreMesh.scale.set(1.6, 1.6, 1.6);
    const el = document.getElementById(`item-${subnode.id}`);
    if (el) el.classList.add("selected");
  }

  translateNode(node, deltaX, deltaY) {
    if (!node) return;
    if (node.isSubnode) {
      // Reposition sub-node in local 3D cluster space
      node.group.position.x += deltaX * 1.5;
      node.group.position.y += deltaY * 1.5;
      return;
    }

    node.lon += deltaX * 45.0;
    node.lat += deltaY * 45.0;
    node.lat = Math.max(-85, Math.min(85, node.lat));

    const newPos = this._latLonToVector3(node.lat, node.lon, this.globe.radius * 1.01);
    node.mesh.position.copy(newPos);
    node.basePos.copy(newPos);
  }

  update(deltaTime) {
    this.nodes.forEach((node) => {
      node.pulseTime += deltaTime * 3.0;
      const scale = 1.0 + Math.sin(node.pulseTime) * 0.25;
      node.ringMesh.scale.set(scale, scale, 1);
      node.ringMesh.material.opacity = 0.8 - (scale - 1.0) * 2.0;
    });

    // Animate sub-nodes expanding smoothly outwards
    this.subnodes.forEach((sub) => {
      sub.currentPos.lerp(sub.targetPos, 0.12);
      sub.group.position.copy(sub.currentPos);
      sub.pulseTime += deltaTime * 4.0;
      const subScale = 1.0 + Math.sin(sub.pulseTime) * 0.2;
      sub.coreMesh.scale.set(subScale, subScale, subScale);
    });
  }
}
