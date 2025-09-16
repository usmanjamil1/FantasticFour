import tkinter as tk
from tkinter import ttk, messagebox
import random
import math
from enum import Enum
from typing import List, Tuple, Dict, Optional
import csv
import os

class CellType(Enum):
    EMPTY = "empty"
    BRIDGE_SITE = "bridge_site"
    HERO = "hero"
    SILVER_SURFER = "silver_surfer"
    GALACTUS = "galactus"
    FRANKLIN = "franklin"
    HEADQUARTERS = "headquarters"

class HeroType(Enum):
    REED_RICHARDS = "Reed Richards"
    SUE_STORM = "Sue Storm"
    JOHNNY_STORM = "Johnny Storm"
    BEN_GRIMM = "Ben Grimm"

class Direction:
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    
    @staticmethod
    def get_neighbors(x: int, y: int, grid_size: int) -> List[Tuple[int, int]]:
        neighbors = []
        for dx, dy in Direction.DIRECTIONS:
            new_x = (x + dx) % grid_size
            new_y = (y + dy) % grid_size
            neighbors.append((new_x, new_y))
        return neighbors

class AStar:
    @staticmethod
    def heuristic(a: Tuple[int, int], b: Tuple[int, int], grid_size: int) -> float:
        # Manhattan distance with wrapping consideration
        dx = min(abs(a[0] - b[0]), grid_size - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), grid_size - abs(a[1] - b[1]))
        return dx + dy
    
    @staticmethod
    def find_path(start: Tuple[int, int], goal: Tuple[int, int], grid_size: int, 
                  obstacles: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: AStar.heuristic(start, goal, grid_size)}
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x[1], float('inf')))[1]
            open_set = [(f, pos) for f, pos in open_set if pos != current]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for neighbor in Direction.get_neighbors(current[0], current[1], grid_size):
                if neighbor in obstacles:
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + AStar.heuristic(neighbor, goal, grid_size)
                    if (f_score[neighbor], neighbor) not in open_set:
                        open_set.append((f_score[neighbor], neighbor))
        
        return []

class BridgeSite:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.health = 100
        self.is_completed = False
        self.under_attack = False
        self.repair_progress = 0
        
    def take_damage(self, damage: int):
        self.health = max(0, self.health - damage)
        if self.health == 0:
            self.is_completed = False
            
    def repair(self, amount: int):
        if self.health < 100:
            self.health = min(100, self.health + amount)
        self.repair_progress += amount
        if self.health >= 100 and self.repair_progress >= 100:
            self.is_completed = True

class Hero:
    def __init__(self, hero_type: HeroType, x: int, y: int):
        self.hero_type = hero_type
        self.x = x
        self.y = y
        self.energy = 100
        self.max_energy = 100
        self.current_task = None
        self.path = []
        self.collaboration_request = None
        
        # Hero-specific abilities
        self.abilities = self._get_abilities()
        
    def _get_abilities(self) -> Dict:
        abilities_map = {
            HeroType.REED_RICHARDS: {
                "strategic_planning": True,
                "threat_prediction": True,
                "energy_cost_multiplier": 1.0,
                "repair_efficiency": 1.2,
                "scan_range": 3
            },
            HeroType.SUE_STORM: {
                "protective_shields": True,
                "stealth": True,
                "energy_cost_multiplier": 0.8,
                "repair_efficiency": 1.0,
                "scan_range": 2
            },
            HeroType.JOHNNY_STORM: {
                "ranged_attack": True,
                "high_mobility": True,
                "energy_cost_multiplier": 1.5,
                "repair_efficiency": 0.8,
                "scan_range": 4
            },
            HeroType.BEN_GRIMM: {
                "close_combat": True,
                "heavy_repair": True,
                "energy_cost_multiplier": 1.2,
                "repair_efficiency": 1.5,
                "scan_range": 1
            }
        }
        return abilities_map.get(self.hero_type, {})
    
    def move(self, new_x: int, new_y: int, grid_size: int) -> bool:
        if self.energy <= 0:
            return False
            
        energy_cost = 5 * self.abilities.get("energy_cost_multiplier", 1.0)
        if self.energy >= energy_cost:
            self.x = new_x % grid_size
            self.y = new_y % grid_size
            self.energy -= energy_cost
            return True
        return False
    
    def scan_area(self, grid_size: int) -> List[Tuple[int, int]]:
        scan_range = self.abilities.get("scan_range", 2)
        scanned_cells = []
        
        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                if dx == 0 and dy == 0:
                    continue
                new_x = (self.x + dx) % grid_size
                new_y = (self.y + dy) % grid_size
                scanned_cells.append((new_x, new_y))
                
        return scanned_cells
    
    def repair_bridge(self, bridge: BridgeSite) -> int:
        if self.energy <= 10:
            return 0
            
        repair_amount = 15 * self.abilities.get("repair_efficiency", 1.0)
        self.energy -= 10
        bridge.repair(repair_amount)
        return repair_amount
    
    def attack_enemy(self, enemy_pos: Tuple[int, int], grid_size: int) -> bool:
        if self.energy <= 15:
            return False
            
        distance = AStar.heuristic((self.x, self.y), enemy_pos, grid_size)
        
        # Johnny can attack at range
        if self.hero_type == HeroType.JOHNNY_STORM and distance <= 3:
            self.energy -= 20
            return True
        # Ben specializes in close combat
        elif self.hero_type == HeroType.BEN_GRIMM and distance <= 1:
            self.energy -= 15
            return True
        # Others need to be adjacent
        elif distance <= 1:
            self.energy -= 15
            return True
            
        return False
    
    def share_energy(self, other_hero: 'Hero', amount: int):
        if self.energy >= amount + 20:  # Keep some energy for self
            transfer = min(amount, other_hero.max_energy - other_hero.energy)
            self.energy -= transfer
            other_hero.energy += transfer

class SilverSurfer:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.energy = 100
        self.max_energy = 100
        self.retreat_threshold = 20
        self.target = None
        self.retreat_mode = False
        
    def move(self, new_x: int, new_y: int, grid_size: int) -> bool:
        self.x = new_x % grid_size
        self.y = new_y % grid_size
        return True
    
    def attack_bridge(self, bridge: BridgeSite) -> bool:
        if self.energy >= 15:
            damage = random.randint(20, 40)
            bridge.take_damage(damage)
            bridge.under_attack = True
            self.energy -= 15
            return True
        return False
    
    def should_retreat(self) -> bool:
        return self.energy < self.retreat_threshold or self.retreat_mode

class GalactusProjection:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.destruction_radius = 1
        self.move_counter = 0
        self.move_frequency = 3  # Moves every 3 turns
        
    def move_towards_target(self, target_pos: Tuple[int, int], grid_size: int):
        if self.move_counter < self.move_frequency:
            self.move_counter += 1
            return
            
        self.move_counter = 0
        
        # Simple movement towards target
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        
        # Handle wrapping
        if abs(dx) > grid_size // 2:
            dx = -dx // abs(dx) if dx != 0 else 0
        if abs(dy) > grid_size // 2:
            dy = -dy // abs(dy) if dy != 0 else 0
            
        if dx != 0:
            self.x = (self.x + (1 if dx > 0 else -1)) % grid_size
        elif dy != 0:
            self.y = (self.y + (1 if dy > 0 else -1)) % grid_size

class DefenseGrid:
    def __init__(self, size: int = 25):
        self.size = size
        self.grid = [[CellType.EMPTY for _ in range(size)] for _ in range(size)]
        
        # Initialize game entities
        self.heroes: List[Hero] = []
        self.bridges: List[BridgeSite] = []
        self.silver_surfer: Optional[SilverSurfer] = None
        self.galactus: Optional[GalactusProjection] = None
        self.franklin_pos: Optional[Tuple[int, int]] = None
        self.headquarters_pos: Tuple[int, int] = (size // 2, size // 2)
        
        # Game state
        self.simulation_step = 0
        self.mission_status = "In Progress"
        self.bridges_completed = 0
        self.total_bridges = 0
        
        self._initialize_game()
        
    def _initialize_game(self):
        # Place headquarters
        hq_x, hq_y = self.headquarters_pos
        self.grid[hq_y][hq_x] = CellType.HEADQUARTERS
        
        # Place Franklin Richards randomly
        franklin_x = random.randint(0, self.size - 1)
        franklin_y = random.randint(0, self.size - 1)
        while (franklin_x, franklin_y) == self.headquarters_pos:
            franklin_x = random.randint(0, self.size - 1)
            franklin_y = random.randint(0, self.size - 1)
        self.franklin_pos = (franklin_x, franklin_y)
        self.grid[franklin_y][franklin_x] = CellType.FRANKLIN
        
        # Create Fantastic Four heroes near headquarters
        hero_types = list(HeroType)
        for i, hero_type in enumerate(hero_types):
            hero_x = (hq_x + (i % 2) * 2 - 1) % self.size
            hero_y = (hq_y + (i // 2) * 2 - 1) % self.size
            hero = Hero(hero_type, hero_x, hero_y)
            self.heroes.append(hero)
            self.grid[hero_y][hero_x] = CellType.HERO
        
        # Create bridge sites randomly
        num_bridges = random.randint(8, 12)
        self.total_bridges = num_bridges
        
        for _ in range(num_bridges):
            bridge_x = random.randint(0, self.size - 1)
            bridge_y = random.randint(0, self.size - 1)
            
            # Ensure bridge doesn't overlap with existing entities
            while (self.grid[bridge_y][bridge_x] != CellType.EMPTY or 
                   AStar.heuristic((bridge_x, bridge_y), self.headquarters_pos, self.size) < 3):
                bridge_x = random.randint(0, self.size - 1)
                bridge_y = random.randint(0, self.size - 1)
            
            bridge = BridgeSite(bridge_x, bridge_y)
            self.bridges.append(bridge)
            self.grid[bridge_y][bridge_x] = CellType.BRIDGE_SITE
        
        # Silver Surfer appears after some time
        if random.random() < 0.3:  # 30% chance to start with Silver Surfer
            self._spawn_silver_surfer()
    
    def _spawn_silver_surfer(self):
        surfer_x = random.randint(0, self.size - 1)
        surfer_y = random.randint(0, self.size - 1)
        while self.grid[surfer_y][surfer_x] != CellType.EMPTY:
            surfer_x = random.randint(0, self.size - 1)
            surfer_y = random.randint(0, self.size - 1)
        
        self.silver_surfer = SilverSurfer(surfer_x, surfer_y)
        self.grid[surfer_y][surfer_x] = CellType.SILVER_SURFER
    
    def _spawn_galactus(self):
        galactus_x = random.randint(0, self.size - 1)
        galactus_y = random.randint(0, self.size - 1)
        while self.grid[galactus_y][galactus_x] != CellType.EMPTY:
            galactus_x = random.randint(0, self.size - 1)
            galactus_y = random.randint(0, self.size - 1)
        
        self.galactus = GalactusProjection(galactus_x, galactus_y)
        self.grid[galactus_y][galactus_x] = CellType.GALACTUS
    
    def get_nearest_bridge_needing_repair(self, hero_pos: Tuple[int, int]) -> Optional[BridgeSite]:
        incomplete_bridges = [b for b in self.bridges if not b.is_completed]
        if not incomplete_bridges:
            return None
        
        def bridge_priority(bridge):
            distance = AStar.heuristic(hero_pos, (bridge.x, bridge.y), self.size)
            urgency = (100 - bridge.health) / 100
            return distance - (urgency * 5)  # Prioritize damaged bridges
        
        return min(incomplete_bridges, key=bridge_priority)
    
    def hero_ai_decision(self, hero: Hero) -> str:
        """Advanced AI decision making for heroes"""
        
        # Check if hero needs energy
        if hero.energy < 30:
            # Find nearest ally to get energy from
            for other_hero in self.heroes:
                if (other_hero != hero and other_hero.energy > 50 and
                    AStar.heuristic((hero.x, hero.y), (other_hero.x, other_hero.y), self.size) <= 2):
                    other_hero.share_energy(hero, 30)
                    return "received_energy"
            
            # Move towards headquarters for recharge
            if (hero.x, hero.y) != self.headquarters_pos:
                path = AStar.find_path((hero.x, hero.y), self.headquarters_pos, self.size, [])
                if path and len(path) > 0:
                    next_pos = path[0]
                    if hero.move(next_pos[0], next_pos[1], self.size):
                        self._update_hero_position(hero, next_pos[0], next_pos[1])
                        return "moving_to_hq"
            else:
                hero.energy = min(hero.max_energy, hero.energy + 20)  # Recharge at HQ
                return "recharging"
        
        # Scan for threats
        scanned_cells = hero.scan_area(self.size)
        threats = []
        
        if self.silver_surfer:
            surfer_pos = (self.silver_surfer.x, self.silver_surfer.y)
            if surfer_pos in scanned_cells:
                threats.append(("silver_surfer", surfer_pos))
        
        if self.galactus:
            galactus_pos = (self.galactus.x, self.galactus.y)
            if galactus_pos in scanned_cells:
                threats.append(("galactus", galactus_pos))
        
        # React to threats based on hero type
        if threats:
            for threat_type, threat_pos in threats:
                if threat_type == "silver_surfer":
                    if hero.hero_type == HeroType.JOHNNY_STORM:
                        if hero.attack_enemy(threat_pos, self.size):
                            return "attacking_silver_surfer"
                    elif hero.hero_type == HeroType.SUE_STORM:
                        # Sue creates protective shields around nearby bridges
                        for bridge in self.bridges:
                            if AStar.heuristic((hero.x, hero.y), (bridge.x, bridge.y), self.size) <= 2:
                                bridge.health = min(100, bridge.health + 10)  # Shield protection
                        return "creating_shields"
                elif threat_type == "galactus":
                    # All heroes should avoid Galactus
                    safe_positions = self._find_safe_positions(hero, threat_pos)
                    if safe_positions:
                        safe_pos = safe_positions[0]
                        if hero.move(safe_pos[0], safe_pos[1], self.size):
                            self._update_hero_position(hero, safe_pos[0], safe_pos[1])
                            return "fleeing_galactus"
        
        # Find and repair bridges
        target_bridge = self.get_nearest_bridge_needing_repair((hero.x, hero.y))
        if target_bridge:
            bridge_pos = (target_bridge.x, target_bridge.y)
            
            # If already at bridge, repair it
            if (hero.x, hero.y) == bridge_pos:
                repair_amount = hero.repair_bridge(target_bridge)
                if repair_amount > 0:
                    return f"repaired_bridge_{repair_amount}"
            else:
                # Move towards bridge
                obstacles = self._get_current_obstacles()
                path = AStar.find_path((hero.x, hero.y), bridge_pos, self.size, obstacles)
                if path and len(path) > 0:
                    next_pos = path[0]
                    if hero.move(next_pos[0], next_pos[1], self.size):
                        self._update_hero_position(hero, next_pos[0], next_pos[1])
                        return "moving_to_bridge"
        
        # Patrol if nothing else to do
        return self._patrol_behavior(hero)
    
    def _find_safe_positions(self, hero: Hero, threat_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        safe_positions = []
        current_distance = AStar.heuristic((hero.x, hero.y), threat_pos, self.size)
        
        for neighbor in Direction.get_neighbors(hero.x, hero.y, self.size):
            new_distance = AStar.heuristic(neighbor, threat_pos, self.size)
            if new_distance > current_distance and self.grid[neighbor[1]][neighbor[0]] == CellType.EMPTY:
                safe_positions.append(neighbor)
        
        return safe_positions
    
    def _patrol_behavior(self, hero: Hero) -> str:
        # Simple patrol: move to a random adjacent empty cell
        neighbors = Direction.get_neighbors(hero.x, hero.y, self.size)
        empty_neighbors = [pos for pos in neighbors if self.grid[pos[1]][pos[0]] == CellType.EMPTY]
        
        if empty_neighbors:
            next_pos = random.choice(empty_neighbors)
            if hero.move(next_pos[0], next_pos[1], self.size):
                self._update_hero_position(hero, next_pos[0], next_pos[1])
                return "patrolling"
        
        return "idle"
    
    def _get_current_obstacles(self) -> List[Tuple[int, int]]:
        obstacles = []
        for y in range(self.size):
            for x in range(self.size):
                if (self.grid[y][x] in [CellType.HERO, CellType.SILVER_SURFER, 
                                       CellType.GALACTUS, CellType.HEADQUARTERS]):
                    obstacles.append((x, y))
        return obstacles
    
    def _update_hero_position(self, hero: Hero, new_x: int, new_y: int):
        # Clear old position
        self.grid[hero.y][hero.x] = CellType.EMPTY
        # Set new position
        hero.x, hero.y = new_x, new_y
        self.grid[new_y][new_x] = CellType.HERO
    
    def silver_surfer_ai(self):
        if not self.silver_surfer:
            return
        
        if self.silver_surfer.should_retreat():
            # Move towards edge of map to retreat
            edge_positions = [
                (0, self.silver_surfer.y), (self.size-1, self.silver_surfer.y),
                (self.silver_surfer.x, 0), (self.silver_surfer.x, self.size-1)
            ]
            target = min(edge_positions, key=lambda pos: AStar.heuristic(
                (self.silver_surfer.x, self.silver_surfer.y), pos, self.size))
            
            path = AStar.find_path((self.silver_surfer.x, self.silver_surfer.y), 
                                 target, self.size, self._get_current_obstacles())
            if path:
                next_pos = path[0]
                self.grid[self.silver_surfer.y][self.silver_surfer.x] = CellType.EMPTY
                self.silver_surfer.move(next_pos[0], next_pos[1], self.size)
                self.grid[self.silver_surfer.y][self.silver_surfer.x] = CellType.SILVER_SURFER
        else:
            # Target nearest incomplete bridge
            incomplete_bridges = [b for b in self.bridges if not b.is_completed]
            if incomplete_bridges:
                target_bridge = min(incomplete_bridges, 
                                  key=lambda b: AStar.heuristic(
                                      (self.silver_surfer.x, self.silver_surfer.y), 
                                      (b.x, b.y), self.size))
                
                if (self.silver_surfer.x, self.silver_surfer.y) == (target_bridge.x, target_bridge.y):
                    self.silver_surfer.attack_bridge(target_bridge)
                else:
                    # Move towards target bridge
                    dx = target_bridge.x - self.silver_surfer.x
                    dy = target_bridge.y - self.silver_surfer.y
                    
                    # Handle wrapping
                    if abs(dx) > self.size // 2:
                        dx = -dx // abs(dx) if dx != 0 else 0
                    if abs(dy) > self.size // 2:
                        dy = -dy // abs(dy) if dy != 0 else 0
                    
                    new_x = self.silver_surfer.x
                    new_y = self.silver_surfer.y
                    
                    if dx != 0:
                        new_x = (self.silver_surfer.x + (1 if dx > 0 else -1)) % self.size
                    elif dy != 0:
                        new_y = (self.silver_surfer.y + (1 if dy > 0 else -1)) % self.size
                    
                    if self.grid[new_y][new_x] in [CellType.EMPTY, CellType.BRIDGE_SITE]:
                        self.grid[self.silver_surfer.y][self.silver_surfer.x] = CellType.EMPTY
                        self.silver_surfer.move(new_x, new_y, self.size)
                        self.grid[self.silver_surfer.y][self.silver_surfer.x] = CellType.SILVER_SURFER
    
    def galactus_ai(self):
        if not self.galactus:
            return
        
        # Target either Franklin or largest cluster of active bridges
        franklin_distance = AStar.heuristic(
            (self.galactus.x, self.galactus.y), self.franklin_pos, self.size)
        
        active_bridges = [b for b in self.bridges if b.is_completed]
        if active_bridges and franklin_distance > 5:
            # Target bridge cluster
            cluster_center = self._find_bridge_cluster_center()
            self.galactus.move_towards_target(cluster_center, self.size)
        else:
            # Target Franklin
            self.galactus.move_towards_target(self.franklin_pos, self.size)
        
        # Update Galactus position
        self.grid[self.galactus.y][self.galactus.x] = CellType.GALACTUS
        
        # Destroy anything in path
        self._galactus_destruction()
    
    def _find_bridge_cluster_center(self) -> Tuple[int, int]:
        active_bridges = [b for b in self.bridges if b.is_completed]
        if not active_bridges:
            return self.franklin_pos
        
        avg_x = sum(b.x for b in active_bridges) / len(active_bridges)
        avg_y = sum(b.y for b in active_bridges) / len(active_bridges)
        return (int(avg_x), int(avg_y))
    
    def _galactus_destruction(self):
        destruction_cells = []
        for dx in range(-self.galactus.destruction_radius, self.galactus.destruction_radius + 1):
            for dy in range(-self.galactus.destruction_radius, self.galactus.destruction_radius + 1):
                x = (self.galactus.x + dx) % self.size
                y = (self.galactus.y + dy) % self.size
                destruction_cells.append((x, y))
        
        for x, y in destruction_cells:
            if self.grid[y][x] == CellType.BRIDGE_SITE:
                # Find and destroy bridge
                for bridge in self.bridges:
                    if bridge.x == x and bridge.y == y:
                        bridge.health = 0
                        bridge.is_completed = False
            elif self.grid[y][x] == CellType.HERO:
                # Find and damage hero
                for hero in self.heroes:
                    if hero.x == x and hero.y == y:
                        hero.energy = max(0, hero.energy - 50)
            elif (x, y) == self.franklin_pos:
                self.mission_status = "FAILED - Franklin captured!"
                return
            
            if self.grid[y][x] != CellType.GALACTUS:
                self.grid[y][x] = CellType.EMPTY
    
    def simulate_step(self):
        self.simulation_step += 1
        
        # Spawn Silver Surfer after 10 steps
        if self.simulation_step == 10 and not self.silver_surfer:
            self._spawn_silver_surfer()
        
        # Spawn Galactus after 25 steps
        if self.simulation_step == 25 and not self.galactus:
            self._spawn_galactus()
        
        # Hero actions
        for hero in self.heroes:
            if hero.energy > 0:
                action = self.hero_ai_decision(hero)
        
        # Silver Surfer actions
        if self.silver_surfer:
            self.silver_surfer_ai()
        
        # Galactus actions
        if self.galactus:
            self.galactus_ai()
        
        # Check win condition
        completed_bridges = sum(1 for b in self.bridges if b.is_completed)
        self.bridges_completed = completed_bridges
        
        if completed_bridges == self.total_bridges:
            self.mission_status = "SUCCESS - All bridges completed!"
        
        # Update bridge states
        for bridge in self.bridges:
            bridge.under_attack = False

class FantasticFourGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fantastic Four: Earth Defense System")
        self.root.geometry("1000x800")
        
        self.defense_grid = DefenseGrid()
        self.cell_size = 20
        self.colors = {
            CellType.EMPTY: "white",
            CellType.BRIDGE_SITE: "blue",
            CellType.HERO: "green", 
            CellType.SILVER_SURFER: "silver",
            CellType.GALACTUS: "purple",
            CellType.FRANKLIN: "gold",
            CellType.HEADQUARTERS: "red"
        }
        
        self.setup_ui()
        self.running = False
        self.update_display()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for grid
        self.canvas = tk.Canvas(main_frame, 
                               width=self.defense_grid.size * self.cell_size,
                               height=self.defense_grid.size * self.cell_size,
                               bg="white")
        self.canvas.pack(side=tk.LEFT, padx=(0, 10))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mission status
        self.status_label = ttk.Label(control_frame, text="Mission Status: In Progress", 
                                     font=("Arial", 12, "bold"))
        self.status_label.pack(pady=5)
        
        # Statistics
        stats_frame = ttk.LabelFrame(control_frame, text="Mission Statistics")
        stats_frame.pack(fill=tk.X, pady=5)
        
        self.step_label = ttk.Label(stats_frame, text="Step: 0")
        self.step_label.pack(anchor=tk.W)
        
        self.bridge_label = ttk.Label(stats_frame, text="Bridges: 0/0")
        self.bridge_label.pack(anchor=tk.W)
        
        self.hero_energy_frame = ttk.LabelFrame(control_frame, text="Hero Status")
        self.hero_energy_frame.pack(fill=tk.X, pady=5)
        
        # Controls
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", 
                                      command=self.start_simulation)
        self.start_button.pack(fill=tk.X, pady=2)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Simulation", 
                                     command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(fill=tk.X, pady=2)
        
        self.step_button = ttk.Button(button_frame, text="Single Step", 
                                     command=self.single_step)
        self.step_button.pack(fill=tk.X, pady=2)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", 
                                      command=self.reset_simulation)
        self.reset_button.pack(fill=tk.X, pady=2)
        
        # Legend
        legend_frame = ttk.LabelFrame(control_frame, text="Legend")
        legend_frame.pack(fill=tk.X, pady=5)
        
        legend_items = [
            ("Hero", "green"),
            ("Bridge Site", "blue"),
            ("Silver Surfer", "silver"),
            ("Galactus", "purple"),
            ("Franklin", "gold"),
            ("Headquarters", "red")
        ]
        
        for item, color in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(fill=tk.X)
            
            color_label = tk.Label(item_frame, bg=color, width=3)
            color_label.pack(side=tk.LEFT, padx=2)
            
            text_label = ttk.Label(item_frame, text=item)
            text_label.pack(side=tk.LEFT)
    
    def update_display(self):
        self.canvas.delete("all")
        
        # Draw grid
        for y in range(self.defense_grid.size):
            for x in range(self.defense_grid.size):
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                cell_type = self.defense_grid.grid[y][x]
                color = self.colors.get(cell_type, "white")
                
                # Special coloring for bridges based on health
                if cell_type == CellType.BRIDGE_SITE:
                    bridge = next((b for b in self.defense_grid.bridges 
                                 if b.x == x and b.y == y), None)
                    if bridge:
                        if bridge.is_completed:
                            color = "darkblue"
                        elif bridge.health < 50:
                            color = "lightcoral"
                        elif bridge.under_attack:
                            color = "orange"
                
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Add text for special cells
                if cell_type == CellType.HERO:
                    hero = next((h for h in self.defense_grid.heroes 
                               if h.x == x and h.y == y), None)
                    if hero:
                        initial = hero.hero_type.value[0]
                        self.canvas.create_text(x1 + self.cell_size//2, y1 + self.cell_size//2,
                                              text=initial, fill="white", font=("Arial", 8, "bold"))
                
                elif cell_type == CellType.FRANKLIN:
                    self.canvas.create_text(x1 + self.cell_size//2, y1 + self.cell_size//2,
                                          text="F", fill="black", font=("Arial", 8, "bold"))
                
                elif cell_type == CellType.HEADQUARTERS:
                    self.canvas.create_text(x1 + self.cell_size//2, y1 + self.cell_size//2,
                                          text="HQ", fill="white", font=("Arial", 6, "bold"))
        
        # Update statistics
        self.step_label.config(text=f"Step: {self.defense_grid.simulation_step}")
        self.bridge_label.config(text=f"Bridges: {self.defense_grid.bridges_completed}/{self.defense_grid.total_bridges}")
        self.status_label.config(text=f"Mission Status: {self.defense_grid.mission_status}")
        
        # Update hero energy display
        for widget in self.hero_energy_frame.winfo_children():
            widget.destroy()
        
        for hero in self.defense_grid.heroes:
            energy_frame = ttk.Frame(self.hero_energy_frame)
            energy_frame.pack(fill=tk.X)
            
            name_label = ttk.Label(energy_frame, text=hero.hero_type.value[:4])
            name_label.pack(side=tk.LEFT)
            
            energy_bar = ttk.Progressbar(energy_frame, length=100, mode='determinate')
            energy_bar['value'] = hero.energy
            energy_bar.pack(side=tk.LEFT, padx=5)
            
            energy_text = ttk.Label(energy_frame, text=f"{hero.energy}%")
            energy_text.pack(side=tk.LEFT)
    
    def start_simulation(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.run_simulation()
    
    def stop_simulation(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def single_step(self):
        self.defense_grid.simulate_step()
        self.update_display()
        
        if "SUCCESS" in self.defense_grid.mission_status or "FAILED" in self.defense_grid.mission_status:
            self.stop_simulation()
            messagebox.showinfo("Mission Complete", self.defense_grid.mission_status)
    
    def run_simulation(self):
        if self.running:
            self.single_step()
            
            if self.running:  # Check if still running after step
                self.root.after(500, self.run_simulation)  # 500ms delay
    
    def reset_simulation(self):
        self.stop_simulation()
        self.defense_grid = DefenseGrid()
        self.update_display()

def save_simulation_data(defense_grid: DefenseGrid, filename: str):
    """Save simulation data to CSV for analysis"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['step', 'hero_name', 'hero_x', 'hero_y', 'hero_energy', 
                     'bridges_completed', 'mission_status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for hero in defense_grid.heroes:
            writer.writerow({
                'step': defense_grid.simulation_step,
                'hero_name': hero.hero_type.value,
                'hero_x': hero.x,
                'hero_y': hero.y,
                'hero_energy': hero.energy,
                'bridges_completed': defense_grid.bridges_completed,
                'mission_status': defense_grid.mission_status
            })

def main():
    """Main function to run the Fantastic Four Defense System"""
    root = tk.Tk()
    app = FantasticFourGUI(root)
    
    # Add menu for saving data
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    file_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    
    def save_data():
        filename = f"simulation_data_step_{app.defense_grid.simulation_step}.csv"
        save_simulation_data(app.defense_grid, filename)
        messagebox.showinfo("Data Saved", f"Simulation data saved to {filename}")
    
    file_menu.add_command(label="Save Simulation Data", command=save_data)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    
    root.mainloop()

if __name__ == "__main__":
    main()