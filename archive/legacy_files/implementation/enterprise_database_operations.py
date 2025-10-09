#!/usr/bin/env python3
"""
Enterprise Database Operations - High-Performance Database Layer
Optimized database operations with caching, connection pooling, and analytics
"""

import json
import logging
import sqlite3
import uuid
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    total_queries: int = 0
    average_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    connection_pool_size: int = 0
    active_connections: int = 0
    slow_queries: int = 0
    error_count: int = 0

class EnterpriseDatabaseOperations:
    """High-performance database operations with enterprise features"""
    
    def __init__(self, 
                 db_path: str,
                 connection_pool_size: int = 20,
                 cache_size: int = 1000,
                 slow_query_threshold: float = 1.0):
        """Initialize enterprise database operations"""
        self.db_path = db_path
        self.connection_pool_size = connection_pool_size
        self.cache_size = cache_size
        self.slow_query_threshold = slow_query_threshold
        
        # Connection pool management
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.active_connections = set()
        
        # Caching system
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = {}  # Time-to-live for cached items
        
        # Performance metrics
        self.metrics = DatabaseMetrics()
        self.metrics_lock = threading.Lock()
        
        # Query optimization
        self.prepared_statements = {}
        self.query_stats = {}
        
        # Initialize systems
        self._initialize_connection_pool()
        self._create_indexes()
        self._optimize_database_settings()
        
        logger.info("✅ Enterprise Database Operations initialized")
        logger.info(f"📊 Pool size: {connection_pool_size}, Cache size: {cache_size}")
    
    def _initialize_connection_pool(self):
        """Initialize optimized connection pool"""
        try:
            for i in range(self.connection_pool_size):
                conn = self._create_optimized_connection()
                self.connection_pool.append(conn)
            
            with self.metrics_lock:
                self.metrics.connection_pool_size = len(self.connection_pool)
            
            logger.info(f"✅ Connection pool initialized: {len(self.connection_pool)} connections")
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            raise
    
    def _create_optimized_connection(self) -> sqlite3.Connection:
        """Create optimized database connection"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Optimize connection settings
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        conn.execute("PRAGMA optimize")
        
        return conn
    
    def _create_indexes(self):
        """Create performance indexes"""
        indexes = [
            # Conversation flow indexes
            "CREATE INDEX IF NOT EXISTS idx_flows_tags ON conversation_flows(flow_tags)",
            "CREATE INDEX IF NOT EXISTS idx_flows_demographics ON conversation_flows(target_demographics)",
            "CREATE INDEX IF NOT EXISTS idx_flows_emotional_range ON conversation_flows(emotional_range_min, emotional_range_max)",
            
            # Node indexes
            "CREATE INDEX IF NOT EXISTS idx_nodes_flow_intensity ON conversation_nodes(flow_id, emotional_intensity)",
            "CREATE INDEX IF NOT EXISTS idx_nodes_context_tags ON conversation_nodes(context_tags)",
            
            # Response indexes
            "CREATE INDEX IF NOT EXISTS idx_responses_personality_score ON assistant_responses(personality_type, empathy_score)",
            "CREATE INDEX IF NOT EXISTS idx_responses_node_personality ON assistant_responses(node_id, personality_type)",
            
            # Transition indexes
            "CREATE INDEX IF NOT EXISTS idx_transitions_condition ON node_transitions(condition_type, probability_weight)",
            "CREATE INDEX IF NOT EXISTS idx_transitions_from_to ON node_transitions(from_node_id, to_node_id)",
            
            # Session indexes
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON conversation_sessions(user_id, start_time)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_state ON conversation_sessions(session_outcome, follow_up_needed)",
            
            # Turn indexes
            "CREATE INDEX IF NOT EXISTS idx_turns_session_time ON conversation_turns(session_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_turns_personality ON conversation_turns(personality_used, emotional_intensity)",
            
            # Analytics indexes
            "CREATE INDEX IF NOT EXISTS idx_analytics_effectiveness ON conversation_analytics(flow_effectiveness_score)",
            "CREATE INDEX IF NOT EXISTS idx_analytics_satisfaction ON conversation_analytics(user_satisfaction_score)",
            
            # Follow-up indexes
            "CREATE INDEX IF NOT EXISTS idx_followup_user_scheduled ON follow_up_triggers(user_id, scheduled_time, completed)",
            "CREATE INDEX IF NOT EXISTS idx_followup_trigger_type ON follow_up_triggers(trigger_type, completed)"
        ]
        
        conn = self._get_connection()
        try:
            for index_sql in indexes:
                conn.execute(index_sql)
            conn.commit()
            logger.info("✅ Performance indexes created")
        except Exception as e:
            logger.error(f"❌ Error creating indexes: {e}")
        finally:
            self._return_connection(conn)
    
    def _optimize_database_settings(self):
        """Optimize database settings for performance"""
        conn = self._get_connection()
        try:
            # Analyze tables for query optimization
            conn.execute("ANALYZE")
            
            # Update statistics
            conn.execute("PRAGMA optimize")
            
            conn.commit()
            logger.info("✅ Database optimized")
        except Exception as e:
            logger.error(f"❌ Error optimizing database: {e}")
        finally:
            self._return_connection(conn)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._return_connection(conn)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get connection from pool with metrics"""
        start_time = time.time()
        
        with self.pool_lock:
            if self.connection_pool:
                conn = self.connection_pool.pop()
                self.active_connections.add(id(conn))
            else:
                # Create new connection if pool is empty
                conn = self._create_optimized_connection()
                self.active_connections.add(id(conn))
        
        with self.metrics_lock:
            self.metrics.active_connections = len(self.active_connections)
        
        return conn
    
    def _return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self.pool_lock:
            self.active_connections.discard(id(conn))
            if len(self.connection_pool) < self.connection_pool_size:
                self.connection_pool.append(conn)
            else:
                conn.close()
        
        with self.metrics_lock:
            self.metrics.active_connections = len(self.active_connections)
    
    def _cache_key(self, query: str, params: tuple = ()) -> str:
        """Generate cache key for query"""
        key_data = f"{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache if valid"""
        with self.cache_lock:
            if cache_key in self.query_cache:
                # Check TTL
                if cache_key in self.cache_ttl:
                    if datetime.now() > self.cache_ttl[cache_key]:
                        # Expired
                        del self.query_cache[cache_key]
                        del self.cache_ttl[cache_key]
                        with self.metrics_lock:
                            self.metrics.cache_misses += 1
                        return None
                
                with self.metrics_lock:
                    self.metrics.cache_hits += 1
                return self.query_cache[cache_key]
            
            with self.metrics_lock:
                self.metrics.cache_misses += 1
            return None
    
    def _store_in_cache(self, cache_key: str, result: Any, ttl_seconds: int = 300):
        """Store result in cache with TTL"""
        with self.cache_lock:
            # Implement LRU eviction if cache is full
            if len(self.query_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
                if oldest_key in self.cache_ttl:
                    del self.cache_ttl[oldest_key]
            
            self.query_cache[cache_key] = result
            self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def _execute_query(self, 
                      query: str, 
                      params: tuple = (), 
                      fetch_one: bool = False,
                      fetch_all: bool = True,
                      use_cache: bool = True,
                      cache_ttl: int = 300) -> Any:
        """Execute query with caching and metrics"""
        start_time = time.time()
        
        # Check cache first
        cache_key = self._cache_key(query, params) if use_cache else None
        if cache_key:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute query
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()
            else:
                result = cursor.rowcount
            
            # Store in cache
            if cache_key and result is not None:
                self._store_in_cache(cache_key, result, cache_ttl)
            
            # Update metrics
            query_time = time.time() - start_time
            self._update_query_metrics(query, query_time)
            
            return result
            
        except Exception as e:
            with self.metrics_lock:
                self.metrics.error_count += 1
            logger.error(f"❌ Query error: {e}")
            raise
        finally:
            self._return_connection(conn)
    
    def _update_query_metrics(self, query: str, query_time: float):
        """Update query performance metrics"""
        with self.metrics_lock:
            self.metrics.total_queries += 1
            
            # Update average query time
            current_avg = self.metrics.average_query_time
            total_queries = self.metrics.total_queries
            self.metrics.average_query_time = (
                (current_avg * (total_queries - 1) + query_time) / total_queries
            )
            
            # Track slow queries
            if query_time > self.slow_query_threshold:
                self.metrics.slow_queries += 1
                logger.warning(f"🐌 Slow query ({query_time:.3f}s): {query[:100]}...")
        
        # Track query statistics
        query_type = query.strip().split()[0].upper()
        if query_type not in self.query_stats:
            self.query_stats[query_type] = {'count': 0, 'total_time': 0.0}
        
        self.query_stats[query_type]['count'] += 1
        self.query_stats[query_type]['total_time'] += query_time
    
    # Conversation Flow Operations
    async def get_conversation_flows(self, 
                                   flow_tags: Optional[List[str]] = None,
                                   emotional_range: Optional[Tuple[int, int]] = None,
                                   target_demographics: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get conversation flows with filtering"""
        
        query = "SELECT * FROM conversation_flows WHERE 1=1"
        params = []
        
        if flow_tags:
            placeholders = ','.join(['?' for _ in flow_tags])
            query += f" AND flow_tags LIKE ANY (VALUES {placeholders})"
            params.extend([f'%{tag}%' for tag in flow_tags])
        
        if emotional_range:
            query += " AND emotional_range_min <= ? AND emotional_range_max >= ?"
            params.extend(emotional_range)
        
        if target_demographics:
            placeholders = ','.join(['?' for _ in target_demographics])
            query += f" AND target_demographics LIKE ANY (VALUES {placeholders})"
            params.extend([f'%{demo}%' for demo in target_demographics])
        
        query += " ORDER BY updated_at DESC"
        
        rows = self._execute_query(query, tuple(params))
        return [dict(row) for row in rows] if rows else []
    
    async def get_conversation_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation node by ID"""
        query = "SELECT * FROM conversation_nodes WHERE node_id = ?"
        row = self._execute_query(query, (node_id,), fetch_one=True)
        return dict(row) if row else None
    
    async def get_starting_node(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Get starting node for a flow"""
        query = """
            SELECT cn.* FROM conversation_nodes cn
            JOIN conversation_flows cf ON cn.node_id = cf.starting_node_id
            WHERE cf.flow_id = ?
        """
        row = self._execute_query(query, (flow_id,), fetch_one=True)
        return dict(row) if row else None
    
    async def get_node_transitions(self, from_node_id: str) -> List[Dict[str, Any]]:
        """Get possible transitions from a node"""
        query = """
            SELECT * FROM node_transitions 
            WHERE from_node_id = ? 
            ORDER BY probability_weight DESC
        """
        rows = self._execute_query(query, (from_node_id,))
        return [dict(row) for row in rows] if rows else []
    
    async def get_next_sequential_node(self, current_node_id: str) -> Optional[Dict[str, Any]]:
        """Get next sequential node in conversation flow"""
        query = """
            SELECT cn.* FROM conversation_nodes cn
            JOIN node_transitions nt ON cn.node_id = nt.to_node_id
            WHERE nt.from_node_id = ? AND nt.condition_type = 'sequential'
            ORDER BY nt.probability_weight DESC
            LIMIT 1
        """
        row = self._execute_query(query, (current_node_id,), fetch_one=True)
        return dict(row) if row else None
    
    # Assistant Response Operations
    async def get_assistant_responses(self, 
                                    node_id: str, 
                                    personality_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get assistant responses for a node"""
        query = "SELECT * FROM assistant_responses WHERE node_id = ?"
        params = [node_id]
        
        if personality_type:
            query += " AND personality_type = ?"
            params.append(personality_type)
        
        query += " ORDER BY empathy_score DESC, naturalness_score DESC"
        
        rows = self._execute_query(query, tuple(params))
        return [dict(row) for row in rows] if rows else []
    
    async def get_best_response_for_personality(self, 
                                              node_id: str, 
                                              personality_type: str) -> Optional[Dict[str, Any]]:
        """Get best response for specific personality type"""
        query = """
            SELECT * FROM assistant_responses 
            WHERE node_id = ? AND personality_type = ?
            ORDER BY (empathy_score + naturalness_score) DESC
            LIMIT 1
        """
        row = self._execute_query(query, (node_id, personality_type), fetch_one=True)
        return dict(row) if row else None
    
    # User Context Operations
    async def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user context"""
        query = "SELECT * FROM user_contexts WHERE user_id = ?"
        row = self._execute_query(query, (user_id,), fetch_one=True)
        return dict(row) if row else None
    
    async def create_user_context(self, user_context: Dict[str, Any]) -> bool:
        """Create new user context"""
        query = """
            INSERT INTO user_contexts 
            (user_id, personality_preference, demographic_info, conversation_preferences)
            VALUES (?, ?, ?, ?)
        """
        params = (
            user_context['user_id'],
            user_context.get('personality_preference', 'gentle_nurturing'),
            json.dumps(user_context.get('demographic_info', {})),
            json.dumps(user_context.get('conversation_preferences', {}))
        )
        
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error creating user context: {e}")
            return False
        finally:
            self._return_connection(conn)
    
    async def update_user_context(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user context"""
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['demographic_info', 'conversation_preferences']:
                set_clauses.append(f"{key} = ?")
                params.append(json.dumps(value))
            else:
                set_clauses.append(f"{key} = ?")
                params.append(value)
        
        if not set_clauses:
            return True
        
        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(user_id)
        
        query = f"UPDATE user_contexts SET {', '.join(set_clauses)} WHERE user_id = ?"
        
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error updating user context: {e}")
            return False
        finally:
            self._return_connection(conn)
    
    # Session Operations
    async def create_conversation_session(self, session_data: Dict[str, Any]) -> bool:
        """Create new conversation session"""
        query = """
            INSERT INTO conversation_sessions 
            (session_id, user_id, flow_id, emotional_intensity_start, session_outcome, follow_up_needed)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        params = (
            session_data['session_id'],
            session_data['user_id'],
            session_data['flow_id'],
            session_data.get('emotional_intensity_start', 5),
            session_data.get('session_outcome', 'active'),
            session_data.get('follow_up_needed', False)
        )
        
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error creating session: {e}")
            return False
        finally:
            self._return_connection(conn)
    
    async def store_conversation_turn(self, session_id: str, turn_data: Dict[str, Any]) -> bool:
        """Store conversation turn"""
        query = """
            INSERT INTO conversation_turns 
            (turn_id, session_id, node_id, turn_number, user_message, assistant_response, 
             personality_used, emotional_intensity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            str(uuid.uuid4()),
            session_id,
            turn_data['node_id'],
            turn_data['turn_number'],
            turn_data['user_message'],
            turn_data['assistant_response'],
            turn_data.get('personality_used', 'gentle_nurturing'),
            turn_data['emotional_intensity']
        )
        
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error storing turn: {e}")
            return False
        finally:
            self._return_connection(conn)
    
    # Analytics Operations
    async def store_conversation_analytics(self, analytics_data: Dict[str, Any]) -> bool:
        """Store conversation analytics"""
        query = """
            INSERT INTO conversation_analytics 
            (analytics_id, session_id, flow_effectiveness_score, user_satisfaction_score,
             conversation_completion_rate, average_response_time, emotional_improvement, personality_match_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            str(uuid.uuid4()),
            analytics_data['session_id'],
            analytics_data.get('flow_effectiveness_score', 0.0),
            analytics_data.get('user_satisfaction_score', 0.0),
            analytics_data.get('conversation_completion_rate', 0.0),
            analytics_data.get('average_response_time', 0.0),
            analytics_data.get('emotional_improvement', 0.0),
            analytics_data.get('personality_match_score', 0.0)
        )
        
        conn = self._get_connection()
        try:
            conn.execute(query, params)
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"❌ Error storing analytics: {e}")
            return False
        finally:
            self._return_connection(conn)
    
    async def store_crisis_event(self, session_id: str, message: str, crisis_level: int) -> bool:
        """Store crisis event for monitoring"""
        # This would typically go to a separate crisis monitoring table
        # For now, we'll log it and could extend the schema later
        logger.critical(f"🚨 CRISIS EVENT - Session: {session_id}, Level: {crisis_level}, Message: {message[:100]}...")
        return True
    
    # Performance and Monitoring
    async def get_database_metrics(self) -> Dict[str, Any]:
        """Get comprehensive database metrics"""
        with self.metrics_lock:
            metrics_dict = asdict(self.metrics)
        
        # Add cache hit rate
        total_cache_requests = self.metrics.cache_hits + self.metrics.cache_misses
        if total_cache_requests > 0:
            metrics_dict['cache_hit_rate'] = self.metrics.cache_hits / total_cache_requests
        else:
            metrics_dict['cache_hit_rate'] = 0.0
        
        # Add query statistics
        metrics_dict['query_stats'] = {}
        for query_type, stats in self.query_stats.items():
            if stats['count'] > 0:
                metrics_dict['query_stats'][query_type] = {
                    'count': stats['count'],
                    'average_time': stats['total_time'] / stats['count']
                }
        
        return metrics_dict
    
    def clear_cache(self):
        """Clear query cache"""
        with self.cache_lock:
            self.query_cache.clear()
            self.cache_ttl.clear()
        logger.info("🧹 Query cache cleared")
    
    def optimize_database(self):
        """Run database optimization"""
        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.execute("PRAGMA optimize")
            conn.commit()
            logger.info("✅ Database optimization complete")
        except Exception as e:
            logger.error(f"❌ Database optimization failed: {e}")
        finally:
            self._return_connection(conn)
    
    def shutdown(self):
        """Graceful shutdown"""
        logger.info("🔄 Shutting down database operations...")
        
        # Close all connections
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
        
        # Clear caches
        self.clear_cache()
        
        logger.info("✅ Database operations shutdown complete")

# Export for use in main engine
__all__ = ['EnterpriseDatabaseOperations', 'DatabaseMetrics']
