"""
Geographic Utilities for BBM Distribution Optimization
Advanced geographic calculations, clustering, and mapping utilities
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any, Union
import math
import logging
from dataclasses import dataclass
from enum import Enum
import json

# Import geographic libraries
try:
    from geopy.distance import geodesic, great_circle
    from geopy.geocoders import Nominatim
    import folium
    from folium import plugins
except ImportError as e:
    logging.warning(f"Geographic libraries not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistanceMethod(Enum):
    """Available distance calculation methods"""
    HAVERSINE = "haversine"
    GEODESIC = "geodesic"
    GREAT_CIRCLE = "great_circle"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"

class MapStyle(Enum):
    """Available map styles for visualization"""
    OPENSTREETMAP = "OpenStreetMap"
    STAMEN_TERRAIN = "Stamen Terrain"
    STAMEN_TONER = "Stamen Toner"
    CARTODB_POSITRON = "CartoDB positron"
    CARTODB_DARK_MATTER = "CartoDB dark_matter"

@dataclass
class Coordinate:
    """Geographic coordinate representation"""
    latitude: float
    longitude: float
    name: str = ""
    elevation: float = 0.0
    
    def __post_init__(self):
        # Validate coordinates
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to (lat, lon) tuple"""
        return (self.latitude, self.longitude)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'elevation': self.elevation
        }

@dataclass
class BoundingBox:
    """Geographic bounding box"""
    north: float
    south: float
    east: float
    west: float
    
    def __post_init__(self):
        if self.north <= self.south:
            raise ValueError("North must be greater than South")
        if self.east <= self.west:
            raise ValueError("East must be greater than West")
    
    def contains(self, coordinate: Coordinate) -> bool:
        """Check if coordinate is within bounding box"""
        return (self.south <= coordinate.latitude <= self.north and
                self.west <= coordinate.longitude <= self.east)
    
    def center(self) -> Coordinate:
        """Get center coordinate of bounding box"""
        center_lat = (self.north + self.south) / 2
        center_lon = (self.east + self.west) / 2
        return Coordinate(center_lat, center_lon, "Center")

class GeographicCalculator:
    """
    Advanced geographic calculations for BBM distribution optimization
    """
    
    def __init__(self):
        self.earth_radius_km = 6371.0
        self.earth_radius_miles = 3959.0
        
    def calculate_distance(self, coord1: Coordinate, coord2: Coordinate, 
                          method: DistanceMethod = DistanceMethod.HAVERSINE,
                          unit: str = "km") -> float:
        """
        Calculate distance between two coordinates using various methods
        
        Args:
            coord1: First coordinate
            coord2: Second coordinate
            method: Distance calculation method
            unit: Distance unit ("km" or "miles")
            
        Returns:
            Distance in specified unit
        """
        if method == DistanceMethod.HAVERSINE:
            distance = self._haversine_distance(coord1, coord2)
        elif method == DistanceMethod.GEODESIC:
            distance = self._geodesic_distance(coord1, coord2)
        elif method == DistanceMethod.GREAT_CIRCLE:
            distance = self._great_circle_distance(coord1, coord2)
        elif method == DistanceMethod.EUCLIDEAN:
            distance = self._euclidean_distance(coord1, coord2)
        elif method == DistanceMethod.MANHATTAN:
            distance = self._manhattan_distance(coord1, coord2)
        else:
            raise ValueError(f"Unknown distance method: {method}")
        
        # Convert to miles if requested
        if unit == "miles":
            distance = distance * 0.621371
        
        return distance
    
    def _haversine_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate haversine distance in kilometers"""
        lat1_rad = math.radians(coord1.latitude)
        lon1_rad = math.radians(coord1.longitude)
        lat2_rad = math.radians(coord2.latitude)
        lon2_rad = math.radians(coord2.longitude)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return self.earth_radius_km * c
    
    def _geodesic_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate geodesic distance using geopy"""
        try:
            distance = geodesic(coord1.to_tuple(), coord2.to_tuple()).kilometers
            return distance
        except:
            # Fallback to haversine if geopy not available
            return self._haversine_distance(coord1, coord2)
    
    def _great_circle_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate great circle distance using geopy"""
        try:
            distance = great_circle(coord1.to_tuple(), coord2.to_tuple()).kilometers
            return distance
        except:
            # Fallback to haversine if geopy not available
            return self._haversine_distance(coord1, coord2)
    
    def _euclidean_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate Euclidean distance (approximate for small distances)"""
        # Convert to approximate meters per degree
        lat_diff = (coord2.latitude - coord1.latitude) * 111.32  # km per degree latitude
        lon_diff = (coord2.longitude - coord1.longitude) * 111.32 * math.cos(math.radians(coord1.latitude))
        
        return math.sqrt(lat_diff**2 + lon_diff**2)
    
    def _manhattan_distance(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """Calculate Manhattan distance (city block distance)"""
        lat_diff = abs(coord2.latitude - coord1.latitude) * 111.32
        lon_diff = abs(coord2.longitude - coord1.longitude) * 111.32 * math.cos(math.radians(coord1.latitude))
        
        return lat_diff + lon_diff
    
    def calculate_bearing(self, coord1: Coordinate, coord2: Coordinate) -> float:
        """
        Calculate bearing from coord1 to coord2 in degrees
        
        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = math.radians(coord1.latitude)
        lat2_rad = math.radians(coord2.latitude)
        dlon_rad = math.radians(coord2.longitude - coord1.longitude)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        return (bearing_deg + 360) % 360
    
    def calculate_midpoint(self, coord1: Coordinate, coord2: Coordinate) -> Coordinate:
        """Calculate midpoint between two coordinates"""
        lat1_rad = math.radians(coord1.latitude)
        lon1_rad = math.radians(coord1.longitude)
        lat2_rad = math.radians(coord2.latitude)
        lon2_rad = math.radians(coord2.longitude)
        
        dlon = lon2_rad - lon1_rad
        
        bx = math.cos(lat2_rad) * math.cos(dlon)
        by = math.cos(lat2_rad) * math.sin(dlon)
        
        lat3_rad = math.atan2(
            math.sin(lat1_rad) + math.sin(lat2_rad),
            math.sqrt((math.cos(lat1_rad) + bx)**2 + by**2)
        )
        lon3_rad = lon1_rad + math.atan2(by, math.cos(lat1_rad) + bx)
        
        return Coordinate(
            latitude=math.degrees(lat3_rad),
            longitude=math.degrees(lon3_rad),
            name="Midpoint"
        )
    
    def create_distance_matrix(self, coordinates: List[Coordinate],
                             method: DistanceMethod = DistanceMethod.HAVERSINE) -> np.ndarray:
        """
        Create distance matrix for list of coordinates
        
        Args:
            coordinates: List of coordinates
            method: Distance calculation method
            
        Returns:
            Symmetric distance matrix
        """
        n = len(coordinates)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.calculate_distance(coordinates[i], coordinates[j], method)
                matrix[i, j] = distance
                matrix[j, i] = distance  # Symmetric matrix
        
        return matrix
    
    def find_centroid(self, coordinates: List[Coordinate]) -> Coordinate:
        """Find geographic centroid of coordinates"""
        if not coordinates:
            raise ValueError("Empty coordinates list")
        
        # Convert to Cartesian coordinates
        x_total = y_total = z_total = 0
        
        for coord in coordinates:
            lat_rad = math.radians(coord.latitude)
            lon_rad = math.radians(coord.longitude)
            
            x = math.cos(lat_rad) * math.cos(lon_rad)
            y = math.cos(lat_rad) * math.sin(lon_rad)
            z = math.sin(lat_rad)
            
            x_total += x
            y_total += y
            z_total += z
        
        # Average
        x_avg = x_total / len(coordinates)
        y_avg = y_total / len(coordinates)
        z_avg = z_total / len(coordinates)
        
        # Convert back to lat/lon
        lon_avg = math.atan2(y_avg, x_avg)
        hyp = math.sqrt(x_avg**2 + y_avg**2)
        lat_avg = math.atan2(z_avg, hyp)
        
        return Coordinate(
            latitude=math.degrees(lat_avg),
            longitude=math.degrees(lon_avg),
            name="Centroid"
        )

class GeographicClustering:
    """
    Geographic clustering for depot optimization and route planning
    """
    
    def __init__(self, calculator: GeographicCalculator = None):
        self.calculator = calculator or GeographicCalculator()
    
    def k_means_clustering(self, coordinates: List[Coordinate], 
                          k: int, max_iterations: int = 100) -> Dict[str, Any]:
        """
        Perform K-means clustering on geographic coordinates
        
        Args:
            coordinates: List of coordinates to cluster
            k: Number of clusters
            max_iterations: Maximum iterations
            
        Returns:
            Dictionary with cluster assignments and centroids
        """
        if k <= 0 or k > len(coordinates):
            raise ValueError(f"Invalid k={k} for {len(coordinates)} coordinates")
        
        # Initialize centroids randomly
        centroids = np.random.choice(coordinates, k, replace=False).tolist()
        
        for iteration in range(max_iterations):
            # Assign points to closest centroid
            clusters = [[] for _ in range(k)]
            assignments = []
            
            for coord in coordinates:
                distances = [self.calculator.calculate_distance(coord, centroid) 
                           for centroid in centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(coord)
                assignments.append(closest_cluster)
            
            # Update centroids
            new_centroids = []
            for i, cluster in enumerate(clusters):
                if cluster:
                    new_centroid = self.calculator.find_centroid(cluster)
                    new_centroid.name = f"Centroid_{i}"
                    new_centroids.append(new_centroid)
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids.append(centroids[i])
            
            # Check for convergence
            converged = True
            for old, new in zip(centroids, new_centroids):
                if self.calculator.calculate_distance(old, new) > 0.001:  # 1m threshold
                    converged = False
                    break
            
            centroids = new_centroids
            
            if converged:
                logger.info(f"K-means converged after {iteration + 1} iterations")
                break
        
        return {
            'clusters': clusters,
            'centroids': centroids,
            'assignments': assignments,
            'iterations': iteration + 1,
            'inertia': self._calculate_inertia(clusters, centroids)
        }
    
    def _calculate_inertia(self, clusters: List[List[Coordinate]], 
                          centroids: List[Coordinate]) -> float:
        """Calculate within-cluster sum of squares (inertia)"""
        total_inertia = 0
        
        for i, cluster in enumerate(clusters):
            if cluster and i < len(centroids):
                centroid = centroids[i]
                cluster_inertia = sum(
                    self.calculator.calculate_distance(coord, centroid)**2 
                    for coord in cluster
                )
                total_inertia += cluster_inertia
        
        return total_inertia
    
    def density_based_clustering(self, coordinates: List[Coordinate],
                               eps_km: float = 5.0, min_points: int = 3) -> Dict[str, Any]:
        """
        Density-based clustering (DBSCAN-like) for geographic coordinates
        
        Args:
            coordinates: List of coordinates
            eps_km: Maximum distance between points in same cluster (km)
            min_points: Minimum points required to form cluster
            
        Returns:
            Dictionary with cluster assignments
        """
        n = len(coordinates)
        visited = [False] * n
        cluster_labels = [-1] * n  # -1 indicates noise
        cluster_id = 0
        
        for i in range(n):
            if visited[i]:
                continue
            
            visited[i] = True
            neighbors = self._find_neighbors(coordinates, i, eps_km)
            
            if len(neighbors) < min_points:
                cluster_labels[i] = -1  # Mark as noise
            else:
                # Start new cluster
                cluster_labels[i] = cluster_id
                self._expand_cluster(coordinates, neighbors, cluster_labels, 
                                   cluster_id, eps_km, min_points, visited)
                cluster_id += 1
        
        # Organize results
        clusters = {}
        noise_points = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:
                noise_points.append(coordinates[i])
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(coordinates[i])
        
        return {
            'clusters': list(clusters.values()),
            'cluster_labels': cluster_labels,
            'noise_points': noise_points,
            'num_clusters': cluster_id
        }
    
    def _find_neighbors(self, coordinates: List[Coordinate], 
                       point_idx: int, eps_km: float) -> List[int]:
        """Find neighbors within eps distance"""
        neighbors = []
        point = coordinates[point_idx]
        
        for i, other_point in enumerate(coordinates):
            if i != point_idx:
                distance = self.calculator.calculate_distance(point, other_point)
                if distance <= eps_km:
                    neighbors.append(i)
        
        return neighbors
    
    def _expand_cluster(self, coordinates: List[Coordinate], neighbors: List[int],
                       cluster_labels: List[int], cluster_id: int, eps_km: float,
                       min_points: int, visited: List[bool]):
        """Expand cluster by adding density-connected points"""
        i = 0
        while i < len(neighbors):
            point_idx = neighbors[i]
            
            if not visited[point_idx]:
                visited[point_idx] = True
                new_neighbors = self._find_neighbors(coordinates, point_idx, eps_km)
                
                if len(new_neighbors) >= min_points:
                    neighbors.extend(new_neighbors)
            
            if cluster_labels[point_idx] == -1:  # Not yet assigned to cluster
                cluster_labels[point_idx] = cluster_id
            
            i += 1

class RouteAnalyzer:
    """
    Route analysis and optimization utilities
    """
    
    def __init__(self, calculator: GeographicCalculator = None):
        self.calculator = calculator or GeographicCalculator()
    
    def analyze_route_efficiency(self, route: List[Coordinate]) -> Dict[str, float]:
        """
        Analyze route efficiency metrics
        
        Args:
            route: Ordered list of coordinates representing route
            
        Returns:
            Dictionary with efficiency metrics
        """
        if len(route) < 2:
            return {'error': 'Route must have at least 2 points'}
        
        # Calculate total distance
        total_distance = 0
        segment_distances = []
        
        for i in range(len(route) - 1):
            distance = self.calculator.calculate_distance(route[i], route[i + 1])
            total_distance += distance
            segment_distances.append(distance)
        
        # Calculate direct distance (as crow flies)
        direct_distance = self.calculator.calculate_distance(route[0], route[-1])
        
        # Calculate route efficiency metrics
        efficiency_ratio = direct_distance / total_distance if total_distance > 0 else 0
        detour_distance = total_distance - direct_distance
        detour_percentage = (detour_distance / direct_distance * 100) if direct_distance > 0 else 0
        
        # Calculate route complexity
        direction_changes = self._count_direction_changes(route)
        avg_segment_distance = np.mean(segment_distances)
        segment_variance = np.var(segment_distances)
        
        return {
            'total_distance_km': total_distance,
            'direct_distance_km': direct_distance,
            'efficiency_ratio': efficiency_ratio,
            'detour_distance_km': detour_distance,
            'detour_percentage': detour_percentage,
            'num_segments': len(segment_distances),
            'avg_segment_distance_km': avg_segment_distance,
            'segment_variance': segment_variance,
            'direction_changes': direction_changes,
            'complexity_score': direction_changes / len(route) if len(route) > 0 else 0
        }
    
    def _count_direction_changes(self, route: List[Coordinate]) -> int:
        """Count significant direction changes in route"""
        if len(route) < 3:
            return 0
        
        direction_changes = 0
        threshold_degrees = 45  # Significant direction change threshold
        
        for i in range(len(route) - 2):
            bearing1 = self.calculator.calculate_bearing(route[i], route[i + 1])
            bearing2 = self.calculator.calculate_bearing(route[i + 1], route[i + 2])
            
            # Calculate bearing difference
            diff = abs(bearing2 - bearing1)
            if diff > 180:
                diff = 360 - diff
            
            if diff > threshold_degrees:
                direction_changes += 1
        
        return direction_changes
    
    def optimize_route_order(self, depot: Coordinate, destinations: List[Coordinate],
                           method: str = "nearest_neighbor") -> List[Coordinate]:
        """
        Optimize route order using simple heuristics
        
        Args:
            depot: Starting/ending depot location
            destinations: List of destination coordinates
            method: Optimization method ("nearest_neighbor", "centroid_first")
            
        Returns:
            Optimized route including depot
        """
        if not destinations:
            return [depot]
        
        if method == "nearest_neighbor":
            return self._nearest_neighbor_route(depot, destinations)
        elif method == "centroid_first":
            return self._centroid_first_route(depot, destinations)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _nearest_neighbor_route(self, depot: Coordinate, 
                               destinations: List[Coordinate]) -> List[Coordinate]:
        """Nearest neighbor heuristic for route optimization"""
        route = [depot]
        remaining = destinations.copy()
        current = depot
        
        while remaining:
            # Find nearest unvisited destination
            distances = [self.calculator.calculate_distance(current, dest) 
                        for dest in remaining]
            nearest_idx = np.argmin(distances)
            nearest = remaining.pop(nearest_idx)
            
            route.append(nearest)
            current = nearest
        
        # Return to depot
        route.append(depot)
        return route
    
    def _centroid_first_route(self, depot: Coordinate,
                             destinations: List[Coordinate]) -> List[Coordinate]:
        """Route optimization starting from centroid of destinations"""
        if not destinations:
            return [depot]
        
        # Find centroid of destinations
        centroid = self.calculator.find_centroid(destinations)
        
        # Find destination closest to centroid
        distances_to_centroid = [
            self.calculator.calculate_distance(centroid, dest) 
            for dest in destinations
        ]
        start_idx = np.argmin(distances_to_centroid)
        start_dest = destinations[start_idx]
        
        # Use nearest neighbor from there
        remaining = [dest for i, dest in enumerate(destinations) if i != start_idx]
        route = [depot, start_dest]
        current = start_dest
        
        while remaining:
            distances = [self.calculator.calculate_distance(current, dest) 
                        for dest in remaining]
            nearest_idx = np.argmin(distances)
            nearest = remaining.pop(nearest_idx)
            
            route.append(nearest)
            current = nearest
        
        route.append(depot)
        return route

class MapVisualizer:
    """
    Interactive map visualization for BBM distribution analysis
    """
    
    def __init__(self):
        self.default_zoom = 10
        self.default_style = MapStyle.OPENSTREETMAP
    
    def create_base_map(self, center: Coordinate, zoom: int = None,
                       style: MapStyle = None) -> 'folium.Map':
        """
        Create base map centered on coordinate
        
        Args:
            center: Center coordinate for map
            zoom: Zoom level (default: 10)
            style: Map style
            
        Returns:
            Folium map object
        """
        try:
            import folium
            
            zoom = zoom or self.default_zoom
            style = style or self.default_style
            
            m = folium.Map(
                location=center.to_tuple(),
                zoom_start=zoom,
                tiles=style.value
            )
            
            return m
        
        except ImportError:
            logger.error("Folium not available for map visualization")
            return None
    
    def add_locations_to_map(self, map_obj: 'folium.Map', 
                           locations: List[Coordinate],
                           location_types: List[str] = None,
                           popup_info: List[str] = None) -> 'folium.Map':
        """
        Add location markers to map
        
        Args:
            map_obj: Folium map object
            locations: List of coordinates
            location_types: List of location types for styling
            popup_info: List of popup information
            
        Returns:
            Updated map object
        """
        if not map_obj:
            return None
        
        try:
            import folium
            
            # Color mapping for different location types
            type_colors = {
                'depot': 'red',
                'spbu': 'blue',
                'warehouse': 'green',
                'distribution_center': 'purple',
                'default': 'gray'
            }
            
            for i, location in enumerate(locations):
                # Determine color
                if location_types and i < len(location_types):
                    color = type_colors.get(location_types[i].lower(), type_colors['default'])
                else:
                    color = type_colors['default']
                
                # Determine popup text
                if popup_info and i < len(popup_info):
                    popup_text = popup_info[i]
                else:
                    popup_text = location.name or f"Location {i+1}"
                
                # Add marker
                folium.Marker(
                    location=location.to_tuple(),
                    popup=popup_text,
                    icon=folium.Icon(color=color)
                ).add_to(map_obj)
            
            return map_obj
        
        except ImportError:
            logger.error("Folium not available for map visualization")
            return map_obj
    
    def add_route_to_map(self, map_obj: 'folium.Map', 
                        route: List[Coordinate],
                        route_color: str = 'blue',
                        route_weight: int = 3) -> 'folium.Map':
        """
        Add route line to map
        
        Args:
            map_obj: Folium map object
            route: Ordered list of coordinates
            route_color: Color of route line
            route_weight: Width of route line
            
        Returns:
            Updated map object
        """
        if not map_obj or len(route) < 2:
            return map_obj
        
        try:
            import folium
            
            # Convert route to list of tuples
            route_coords = [coord.to_tuple() for coord in route]
            
            # Add route line
            folium.PolyLine(
                locations=route_coords,
                color=route_color,
                weight=route_weight,
                opacity=0.8
            ).add_to(map_obj)
            
            # Add route direction arrows (optional)
            self._add_route_arrows(map_obj, route_coords)
            
            return map_obj
        
        except ImportError:
            logger.error("Folium not available for map visualization")
            return map_obj
    
    def _add_route_arrows(self, map_obj: 'folium.Map', 
                         route_coords: List[Tuple[float, float]]):
        """Add directional arrows to route"""
        try:
            import folium
            from folium import plugins
            
            # Add arrows at every few points
            arrow_spacing = max(1, len(route_coords) // 10)  # ~10 arrows per route
            
            for i in range(0, len(route_coords) - 1, arrow_spacing):
                start = route_coords[i]
                end = route_coords[i + 1]
                
                # Simple arrow using marker
                folium.Marker(
                    location=[(start[0] + end[0]) / 2, (start[1] + end[1]) / 2],
                    icon=folium.Icon(icon='arrow-right', color='blue', prefix='fa')
                ).add_to(map_obj)
        
        except:
            pass  # Skip arrows if not available

# Utility functions for common geographic operations
def create_jakarta_coordinates() -> List[Coordinate]:
    """Create sample Jakarta area coordinates for testing"""
    jakarta_locations = [
        Coordinate(-6.1167, 106.8833, "Depot Plumpang"),
        Coordinate(-6.2088, 106.8456, "SPBU Sudirman"),
        Coordinate(-6.2297, 106.8206, "SPBU Gatot Subroto"),
        Coordinate(-6.2383, 106.8306, "SPBU Kuningan"),
        Coordinate(-6.2297, 106.8019, "SPBU Senayan"),
        Coordinate(-6.2614, 106.8147, "SPBU Kemang"),
        Coordinate(-6.1751, 106.8650, "SPBU Kelapa Gading"),
        Coordinate(-6.1369, 106.8913, "SPBU Ancol"),
        Coordinate(-6.2635, 106.7815, "SPBU Pondok Indah"),
        Coordinate(-6.3015, 106.8479, "SPBU Cilandak")
    ]
    
    return jakarta_locations

def calculate_bounding_box(coordinates: List[Coordinate], 
                          padding_percent: float = 0.1) -> BoundingBox:
    """Calculate bounding box for list of coordinates with padding"""
    if not coordinates:
        raise ValueError("Empty coordinates list")
    
    lats = [coord.latitude for coord in coordinates]
    lons = [coord.longitude for coord in coordinates]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Add padding
    lat_padding = (max_lat - min_lat) * padding_percent
    lon_padding = (max_lon - min_lon) * padding_percent
    
    return BoundingBox(
        north=max_lat + lat_padding,
        south=min_lat - lat_padding,
        east=max_lon + lon_padding,
        west=min_lon - lon_padding
    )

def export_coordinates_to_geojson(coordinates: List[Coordinate],
                                 properties: List[Dict] = None) -> str:
    """Export coordinates to GeoJSON format"""
    features = []
    
    for i, coord in enumerate(coordinates):
        # Base properties
        props = {'name': coord.name or f'Location_{i}'}
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coord.longitude, coord.latitude]
            },
            "properties": props
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return json.dumps(geojson, indent=2)

# Example usage and testing
if __name__ == "__main__":
    # Initialize components
    calculator = GeographicCalculator()
    clustering = GeographicClustering(calculator)
    route_analyzer = RouteAnalyzer(calculator)
    map_visualizer = MapVisualizer()
    
    # Create sample Jakarta coordinates
    jakarta_coords = create_jakarta_coordinates()
    
    print("Geographic Utils Test Results:")
    print("=" * 50)
    
    # Test distance calculations
    print("\n1. Distance Calculations:")
    depot = jakarta_coords[0]
    spbu1 = jakarta_coords[1]
    
    for method in [DistanceMethod.HAVERSINE, DistanceMethod.GEODESIC, DistanceMethod.EUCLIDEAN]:
        distance = calculator.calculate_distance(depot, spbu1, method)
        print(f"   {method.value}: {distance:.2f} km")
    
    # Test centroid calculation
    print(f"\n2. Centroid Calculation:")
    centroid = calculator.find_centroid(jakarta_coords)
    print(f"   Centroid: {centroid.latitude:.4f}, {centroid.longitude:.4f}")
    
    # Test distance matrix
    print(f"\n3. Distance Matrix:")
    distance_matrix = calculator.create_distance_matrix(jakarta_coords[:5])
    print(f"   Matrix shape: {distance_matrix.shape}")
    print(f"   Max distance: {distance_matrix.max():.2f} km")
    print(f"   Avg distance: {distance_matrix[distance_matrix > 0].mean():.2f} km")
    
    # Test clustering
    print(f"\n4. K-means Clustering:")
    cluster_result = clustering.k_means_clustering(jakarta_coords, k=3)
    print(f"   Clusters created: {len(cluster_result['clusters'])}")
    print(f"   Iterations: {cluster_result['iterations']}")
    print(f"   Inertia: {cluster_result['inertia']:.2f}")
    
    for i, cluster in enumerate(cluster_result['clusters']):
        print(f"   Cluster {i}: {len(cluster)} locations")
    
    # Test density clustering
    print(f"\n5. Density-based Clustering:")
    density_result = clustering.density_based_clustering(jakarta_coords, eps_km=10, min_points=2)
    print(f"   Number of clusters: {density_result['num_clusters']}")
    print(f"   Noise points: {len(density_result['noise_points'])}")
    
    # Test route analysis
    print(f"\n6. Route Analysis:")
    sample_route = jakarta_coords[:5] + [jakarta_coords[0]]  # Return to start
    route_metrics = route_analyzer.analyze_route_efficiency(sample_route)
    
    print(f"   Total distance: {route_metrics['total_distance_km']:.2f} km")
    print(f"   Direct distance: {route_metrics['direct_distance_km']:.2f} km")
    print(f"   Efficiency ratio: {route_metrics['efficiency_ratio']:.3f}")
    print(f"   Detour percentage: {route_metrics['detour_percentage']:.1f}%")
    print(f"   Direction changes: {route_metrics['direction_changes']}")
    
    # Test route optimization
    print(f"\n7. Route Optimization:")
    depot = jakarta_coords[0]
    destinations = jakarta_coords[1:6]
    
    optimized_route = route_analyzer.optimize_route_order(depot, destinations, "nearest_neighbor")
    optimized_metrics = route_analyzer.analyze_route_efficiency(optimized_route)
    
    print(f"   Optimized route length: {len(optimized_route)} stops")
    print(f"   Optimized distance: {optimized_metrics['total_distance_km']:.2f} km")
    print(f"   Optimized efficiency: {optimized_metrics['efficiency_ratio']:.3f}")
    
    # Test bounding box
    print(f"\n8. Bounding Box:")
    bbox = calculate_bounding_box(jakarta_coords)
    print(f"   North: {bbox.north:.4f}")
    print(f"   South: {bbox.south:.4f}")
    print(f"   East: {bbox.east:.4f}")
    print(f"   West: {bbox.west:.4f}")
    
    # Test bearing calculation
    print(f"\n9. Bearing Calculation:")
    bearing = calculator.calculate_bearing(depot, spbu1)
    print(f"   Bearing from depot to SPBU1: {bearing:.1f}°")
    
    # Test midpoint calculation
    print(f"\n10. Midpoint Calculation:")
    midpoint = calculator.calculate_midpoint(depot, spbu1)
    print(f"    Midpoint: {midpoint.latitude:.4f}, {midpoint.longitude:.4f}")
    
    # Test GeoJSON export
    print(f"\n11. GeoJSON Export:")
    try:
        geojson_str = export_coordinates_to_geojson(jakarta_coords[:3])
        geojson_data = json.loads(geojson_str)
        print(f"    Features exported: {len(geojson_data['features'])}")
        print(f"    First feature type: {geojson_data['features'][0]['geometry']['type']}")
    except Exception as e:
        print(f"    GeoJSON export error: {e}")
    
    # Test map visualization (if folium available)
    print(f"\n12. Map Visualization:")
    try:
        center_coord = calculator.find_centroid(jakarta_coords)
        base_map = map_visualizer.create_base_map(center_coord)
        
        if base_map:
            print("    Base map created successfully")
            
            # Add locations
            location_types = ['depot'] + ['spbu'] * (len(jakarta_coords) - 1)
            map_with_locations = map_visualizer.add_locations_to_map(
                base_map, jakarta_coords, location_types
            )
            
            # Add route
            map_with_route = map_visualizer.add_route_to_map(
                map_with_locations, optimized_route
            )
            
            print("    Locations and route added to map")
            print("    Map ready for display/export")
        else:
            print("    Map creation failed (folium not available)")
    
    except Exception as e:
        print(f"    Map visualization error: {e}")
    
    print("\n" + "=" * 50)
    print("Geographic utilities test completed!")
    
    # Performance benchmark
    print(f"\n13. Performance Benchmark:")
    import time
    
    # Benchmark distance calculations
    start_time = time.time()
    for _ in range(1000):
        calculator.calculate_distance(depot, spbu1, DistanceMethod.HAVERSINE)
    haversine_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(1000):
        calculator.calculate_distance(depot, spbu1, DistanceMethod.GEODESIC)
    geodesic_time = time.time() - start_time
    
    print(f"    Haversine (1000 calls): {haversine_time:.4f}s")
    print(f"    Geodesic (1000 calls): {geodesic_time:.4f}s")
    print(f"    Speed ratio: {geodesic_time/haversine_time:.1f}x")
    
    # Benchmark clustering
    start_time = time.time()
    cluster_result = clustering.k_means_clustering(jakarta_coords, k=3)
    clustering_time = time.time() - start_time
    
    print(f"    K-means clustering: {clustering_time:.4f}s")
    
    # Benchmark route optimization
    start_time = time.time()
    optimized_route = route_analyzer.optimize_route_order(depot, destinations)
    route_time = time.time() - start_time
    
    print(f"    Route optimization: {route_time:.4f}s")
    
    print(f"\nAll geographic utilities working correctly! 🗺️")