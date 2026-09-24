/**
 * Spatial Node Manager handling interactive 3D nodes mounted on the globe.
 */

class SpatialNodeManager {
  constructor(globe) {
    this.globe = globe;
    this.nodes = [];
    this.hoveredNode = null;
    this.selectedNode = null;
    this.manipulatedNode = null;

    this.nodesGroup = new THREE.Group();
    this.globe.globeGroup.add(this.nodesGroup);

    this.raycaster = new THREE.Raycaster();
    this._initSampleNodes();
    this._initConnectionArcs();
  }

  _initSampleNodes() {
    const nodeData = [
      { id: "node-alpha", label: "Alpha Station", lat: 37.77, lon: -122.41, color: 0x00f0ff },
      { id: "node-beta", label: "Beta Outpost", lat: 51.50, lon: -0.12, color: 0x3b82f6 },
      { id: "node-gamma", label: "Gamma Relay", lat: 35.67, lon: 139.65, color: 0x10b981 },
      { id: "node-delta", label: "Delta Hub", lat: -33.86, lon: 151.20, color: 0xa855f7 },
      { id: "node-epsilon", label: "Epsilon Sensor", lat: 1.35, lon: 103.81, color: 0xf59e0b },
      { id: "node-zeta", label: "Zeta Terminal", lat: -22.90, lon: -43.17, color: 0xf43f5e },
    ];

    nodeData.forEach((data) => {
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

    // 1. Core Sphere
    const coreGeo = new THREE.SphereGeometry(0.065, 16, 16);
    const coreMat = new THREE.MeshStandardMaterial({
      color: data.color,
      emissive: data.color,
      emissiveIntensity: 0.8,
      roughness: 0.2,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    group.add(coreMesh);

    // 2. Outer Pulsing Beacon Ring
    const ringGeo = new THREE.RingGeometry(0.08, 0.11, 32);
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
      mesh: group,
      coreMesh: coreMesh,
      ringMesh: ringMesh,
      basePos: pos.clone(),
      isSelected: false,
      isHovered: false,
      pulseTime: Math.random() * Math.PI,
    };
  }

  _initConnectionArcs() {
    // Great-circle spline connection arcs between nodes
    const curveMat = new THREE.LineBasicMaterial({
      color: 0x00f0ff,
      transparent: true,
      opacity: 0.35,
      blending: THREE.AdditiveBlending,
    });

    for (let i = 0; i < this.nodes.length - 1; i++) {
      const p1 = this.nodes[i].basePos;
      const p2 = this.nodes[i + 1].basePos;
      const mid = p1.clone().add(p2).multiplyScalar(0.5).normalize().multiplyScalar(this.globe.radius * 1.35);

      const curve = new THREE.QuadraticBezierCurve3(p1, mid, p2);
      const points = curve.getPoints(32);
      const curveGeo = new THREE.BufferGeometry().setFromPoints(points);
      const curveMesh = new THREE.Line(curveGeo, curveMat);
      this.nodesGroup.add(curveMesh);
    }
  }

  testRaycast(camera, ndcCoords) {
    this.raycaster.setFromCamera(ndcCoords, camera);
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
        this.hoveredNode.coreMesh.material.emissiveIntensity = 0.8;
      }
      this.hoveredNode = newHover;
      if (this.hoveredNode) {
        this.hoveredNode.coreMesh.scale.set(1.4, 1.4, 1.4);
        this.hoveredNode.coreMesh.material.emissiveIntensity = 1.8;
      }
    }

    return this.hoveredNode;
  }

  selectNode(node) {
    if (this.selectedNode) {
      this.selectedNode.isSelected = false;
      this.selectedNode.coreMesh.material.color.setHex(this.selectedNode.color);
    }

    this.selectedNode = node;
    if (this.selectedNode) {
      this.selectedNode.isSelected = true;
      this.selectedNode.coreMesh.material.color.setHex(0xffffff);
      this.selectedNode.coreMesh.scale.set(1.6, 1.6, 1.6);
    }
  }

  translateNode(node, deltaX, deltaY) {
    if (!node) return;
    node.lon += deltaX * 45.0;
    node.lat += deltaY * 45.0;
    node.lat = Math.max(-85, Math.min(85, node.lat));

    const newPos = this._latLonToVector3(node.lat, node.lon, this.globe.radius * 1.01);
    node.mesh.position.copy(newPos);
  }

  update(deltaTime) {
    this.nodes.forEach((node) => {
      node.pulseTime += deltaTime * 3.0;
      const scale = 1.0 + Math.sin(node.pulseTime) * 0.25;
      node.ringMesh.scale.set(scale, scale, 1);
      node.ringMesh.material.opacity = 0.8 - (scale - 1.0) * 2.0;
    });
  }
}
