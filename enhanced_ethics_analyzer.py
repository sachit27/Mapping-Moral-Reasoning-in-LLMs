import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional, Union
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
import networkx as nx
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EnhancedEthicsAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with expanded capabilities for moral framework analysis.
        
        Args:
            data_path: Path to the JSON file containing scenarios and responses
        """
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.scenarios = self.data['scenarios']
        
        # Load NLP model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Attempt to load sentence transformer for embeddings
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.use_embeddings = True
        except:
            print("Warning: SentenceTransformer model not available. Will use simpler text analysis.")
            self.use_embeddings = False
        
        # Initialize expanded dictionaries with hierarchical structure
        self.init_ethical_dictionaries()
        
        # For convenience in referencing top-level categories
        self.ethical_principle_keys = list(self.ethical_principles.keys())
        self.safety_category_keys = list(self.safety_categories.keys())
        
        # Reference statements for semantic similarity
        self.ethical_framework_statements = self.init_ethical_framework_statements()
        
        # Counterfactual variations for scenario-based analysis
        self.counterfactual_variations = self.init_counterfactual_variations()
        
        # Axiom tests for frameworks
        self.axiom_tests = self.init_axiom_tests()

    def init_ethical_dictionaries(self):
        """Initialize the expanded ethical dictionaries with hierarchical structure."""
        self.ethical_principles = {
            'consequentialism': {
                'terms': [
                    'consequence', 'consequences', 'outcome', 'outcomes', 'result', 'results',
                    'impact', 'impacts', 'effect', 'effects', 'benefit', 'benefits', 'harm',
                    'maximize', 'maximization', 'minimize', 'minimization'
                ],
                'sub_principles': {
                    'utilitarianism': [
                        'utility', 'greatest happiness', 'greatest good', 'greater good',
                        'aggregate welfare', 'net benefit', 'cost-benefit', 'overall welfare',
                        'happiness', 'pleasure', 'satisfaction', 'welfare', 'wellbeing'
                    ],
                    'act_utilitarianism': [
                        'individual act', 'specific action', 'case by case', 'particular action',
                        'individual decision'
                    ],
                    'rule_utilitarianism': [
                        'rule utility', 'general rule', 'guideline', 'principle utility',
                        'utility rule', 'utility principle'
                    ],
                    'preference_utilitarianism': [
                        'preference', 'preferences', 'desire', 'desires', 'want', 'wants',
                        'choice', 'choices', 'satisfaction of preferences'
                    ]
                }
            },
            'deontological': {
                'terms': [
                    'duty', 'duties', 'obligation', 'obligations', 'rule', 'rules',
                    'principle', 'principles', 'imperative', 'command', 'normative',
                    'moral law', 'moral rule', 'intrinsic', 'inherent'
                ],
                'sub_principles': {
                    'kantian': [
                        'categorical imperative', 'universal law', 'kingdom of ends',
                        'rational being', 'rationality', 'dignity', 'respect for persons',
                        'humanity as end'
                    ],
                    'divine_command': [
                        'divine', 'god', 'gods', 'religion', 'religious', 'commandment',
                        'sacred', 'holy', 'theological'
                    ],
                    'natural_law': [
                        'natural law', 'nature', 'natural', 'human nature', 'teleology',
                        'purpose', 'function', 'natural purpose'
                    ],
                    'rights_based': [
                        'rights', 'natural rights', 'human rights', 'right', 'entitlement',
                        'claim', 'liberty', 'freedom', 'dignity'
                    ]
                }
            },
            'virtue_ethics': {
                'terms': [
                    'virtue', 'virtues', 'character', 'excellence', 'flourishing', 'eudaimonia',
                    'habit', 'traits', 'disposition', 'moral character', 'integrity'
                ],
                'sub_principles': {
                    'aristotelian': [
                        'mean', 'golden mean', 'moderation', 'balance', 'practical wisdom',
                        'phronesis', 'excellence', 'eudaimonia', 'flourishing'
                    ],
                    'care_ethics': [
                        'care', 'caring', 'empathy', 'compassion', 'nurture', 'relationship',
                        'relationships', 'interpersonal', 'connection', 'attachment'
                    ],
                    'confucian': [
                        'benevolence', 'ren', 'righteousness', 'yi', 'propriety', 'li',
                        'wisdom', 'zhi', 'trustworthiness', 'xin', 'filial piety', 'xiào'
                    ]
                }
            },
            'justice': {
                'terms': [
                    'justice', 'fair', 'fairness', 'equity', 'equal', 'equality', 'inequality',
                    'distribution', 'distributive', 'just', 'impartial', 'unbiased'
                ],
                'sub_principles': {
                    'distributive_justice': [
                        'distribution', 'distribute', 'allocation', 'allocate', 'resources',
                        'goods', 'benefits', 'burdens'
                    ],
                    'procedural_justice': [
                        'procedure', 'process', 'fair process', 'due process', 'just procedure',
                        'fair hearing', 'impartial procedure'
                    ],
                    'social_justice': [
                        'social', 'societal', 'institutional', 'structural', 'systemic',
                        'marginalized', 'disadvantaged', 'oppressed', 'privilege'
                    ],
                    'rawlsian': [
                        'veil of ignorance', 'original position', 'difference principle',
                        'basic structure', 'primary goods', 'public reason', 'reflective equilibrium'
                    ]
                }
            },
            'moral_relativism': {
                'terms': [
                    'relative', 'relativism', 'cultural', 'culture', 'subjective', 'perspective',
                    'context', 'contextual', 'society', 'societal', 'norms', 'customs'
                ],
                'sub_principles': {
                    'cultural_relativism': [
                        'culture', 'cultural', 'society', 'social', 'tradition', 'customs',
                        'practices', 'beliefs', 'cultural context'
                    ],
                    'moral_subjectivism': [
                        'subjective', 'personal', 'individual', 'opinion', 'perspective',
                        'viewpoint', 'belief', 'attitude'
                    ]
                }
            },
            'pragmatism': {
                'terms': [
                    'practical', 'workable', 'pragmatic', 'useful', 'utility', 'experience',
                    'experiential', 'experiment', 'situational', 'context', 'contextual'
                ],
                'sub_principles': {
                    'american_pragmatism': [
                        'experience', 'experiment', 'inquiry', 'problem-solving',
                        'consequences', 'practical'
                    ],
                    'moral_pluralism': [
                        'plurality', 'multiple', 'diverse', 'diversity', 'pluralistic',
                        'many', 'various', 'heterogeneous'
                    ]
                }
            },
            'feminism': {
                'terms': [
                    'feminist', 'feminism', 'gender', 'patriarchy', 'sexism', 'misogyny',
                    'equality', 'oppression', 'marginalization', 'power', 'privilege'
                ],
                'sub_principles': {
                    'liberal_feminism': [
                        'equality', 'equal rights', 'equal opportunity', 'liberty', 'autonomy',
                        'choice', 'freedom'
                    ],
                    'care_feminism': [
                        'care', 'caring', 'relationships', 'interdependence', 'connection',
                        'responsibility', 'attentiveness'
                    ],
                    'intersectional_feminism': [
                        'intersectionality', 'intersection', 'race', 'class', 'sexuality',
                        'ability', 'multiple forms of oppression', 'interconnected'
                    ]
                }
            },
            'autonomy': {
                'terms': [
                    'autonomy', 'autonomous', 'self-determination', 'self-governance',
                    'independence', 'freedom', 'liberty', 'choice', 'consent', 'agency',
                    'self-rule', 'sovereignty'
                ],
                'sub_principles': {
                    'individual_autonomy': [
                        'individual', 'person', 'personal', 'self', 'oneself', 'own decision',
                        'self-direction'
                    ],
                    'informed_consent': [
                        'informed', 'information', 'disclosure', 'understanding', 'voluntary',
                        'capacity', 'competence'
                    ],
                    'bodily_autonomy': [
                        'body', 'bodily', 'physical', 'corporeal', 'somatic', 'integrity'
                    ]
                }
            }
        }
        
        self.safety_categories = {
            'immediate_harm': {
                'terms': [
                    'immediate', 'direct', 'urgent', 'risk', 'danger', 'threat', 'harm',
                    'short-term', 'instantaneous', 'acute', 'injury', 'damage', 'hazard',
                    'emergency', 'crisis', 'peril'
                ],
                'sub_categories': {
                    'physical_harm': [
                        'physical', 'injury', 'pain', 'hurt', 'wound', 'bodily', 'violence',
                        'assault', 'attack', 'trauma'
                    ],
                    'psychological_harm': [
                        'psychological', 'mental', 'emotional', 'distress', 'anguish',
                        'anxiety', 'depression', 'suffering', 'stress'
                    ],
                    'safety_risk': [
                        'unsafe', 'dangerous', 'hazardous', 'precarious', 'risky',
                        'threatening', 'menacing', 'perilous'
                    ]
                }
            },
            'long_term_impact': {
                'terms': [
                    'future', 'long-term', 'sustainable', 'lasting', 'downstream',
                    'ongoing', 'prolonged', 'chronic', 'enduring', 'persistent',
                    'permanent', 'durable', 'lingering', 'sustained'
                ],
                'sub_categories': {
                    'environmental': [
                        'environment', 'nature', 'ecosystem', 'planet', 'climate',
                        'biodiversity', 'habitat', 'sustainability', 'pollution'
                    ],
                    'social_impact': [
                        'society', 'culture', 'community', 'civilization', 'custom',
                        'tradition', 'social fabric', 'social cohesion'
                    ],
                    'intergenerational': [
                        'generation', 'generations', 'future generations', 'children',
                        'descendants', 'legacy', 'inheritance', 'posterity'
                    ]
                }
            },
            'systemic_risk': {
                'terms': [
                    'systemic', 'structural', 'institutional', 'widespread', 'societal',
                    'industry-wide', 'broad impact', 'macro-level', 'collective risk',
                    'cascading', 'network', 'interconnected'
                ],
                'sub_categories': {
                    'infrastructure': [
                        'infrastructure', 'system', 'network', 'backbone', 'framework',
                        'foundation', 'structure', 'platform'
                    ],
                    'governance': [
                        'governance', 'regulation', 'policy', 'law', 'rule', 'standard',
                        'oversight', 'compliance', 'enforcement'
                    ],
                    'economic': [
                        'economic', 'financial', 'market', 'economy', 'fiscal',
                        'monetary', 'trade', 'business', 'commercial'
                    ]
                }
            },
            'individual_protection': {
                'terms': [
                    'individual', 'personal', 'privacy', 'confidential', 'protection',
                    'personal data', 'personal info', 'security', 'safeguard',
                    'shield', 'defend', 'preserve', 'secure', 'guard'
                ],
                'sub_categories': {
                    'privacy': [
                        'privacy', 'private', 'confidential', 'secret', 'hidden',
                        'concealed', 'undisclosed', 'personal information'
                    ],
                    'security': [
                        'security', 'secure', 'protection', 'safeguard', 'safety',
                        'defense', 'shield', 'barrier'
                    ],
                    'rights': [
                        'rights', 'entitlement', 'prerogative', 'privilege', 'liberty',
                        'freedom', 'claim', 'due', 'human rights'
                    ]
                }
            },
            'collective_welfare': {
                'terms': [
                    'community', 'public', 'collective', 'society', 'common good',
                    'communal', 'shared benefit', 'public interest', 'social welfare',
                    'general welfare', 'public good', 'common welfare'
                ],
                'sub_categories': {
                    'public_health': [
                        'health', 'wellbeing', 'wellness', 'disease', 'illness',
                        'epidemic', 'pandemic', 'healthcare', 'medical'
                    ],
                    'social_harmony': [
                        'harmony', 'peace', 'unity', 'solidarity', 'cooperation',
                        'collaboration', 'cohesion', 'togetherness'
                    ],
                    'equity': [
                        'equity', 'fairness', 'justice', 'impartiality', 'equality',
                        'equal', 'unbiased', 'balanced', 'nondiscriminatory'
                    ]
                }
            }
        }

    def init_ethical_framework_statements(self) -> Dict[str, List[str]]:
        """
        Initialize canonical statements for major ethical frameworks
        to use in semantic similarity analysis.
        """
        return {
            'utilitarianism': [
                "The right action is the one that produces the greatest amount of good.",
                "We should maximize overall happiness and well-being.",
                "The morality of an action depends on its consequences, not intentions.",
                "The ends justify the means if they produce the greatest good for the greatest number."
            ],
            'kantian_deontology': [
                "Always act according to that maxim which you can will as a universal law.",
                "Treat people as ends in themselves, never merely as means.",
                "The moral worth of an action lies in the intention, not the consequences.",
                "Some actions are wrong regardless of their outcomes."
            ],
            'virtue_ethics': [
                "A virtuous person will naturally make the right decisions.",
                "We should develop excellent character traits like courage, honesty, and compassion.",
                "The right action is what a virtuous person would do in the circumstances.",
                "Moral development is about cultivating virtues through practice and habit."
            ],
            'care_ethics': [
                "Moral decisions should maintain relationships and respond to needs.",
                "Empathy and care for others should guide our actions.",
                "Ethics is about responding to particular others in their concrete circumstances.",
                "Moral decisions should consider networks of relationships and dependencies."
            ],
            'rawlsian_justice': [
                "Just principles are those that would be chosen behind a veil of ignorance.",
                "Social and economic inequalities must benefit the least advantaged.",
                "Each person should have equal basic liberties compatible with liberty for all.",
                "Justice is fairness in the distribution of rights, opportunities, and resources."
            ],
            'moral_relativism': [
                "Moral standards vary across cultures and have no objective validity.",
                "What is right or wrong depends on cultural context and social norms.",
                "There are no universal moral principles that apply to all societies.",
                "Morality is determined by what is accepted within a particular society."
            ],
            'pragmatic_ethics': [
                "The right course of action is what works in practice to solve problems.",
                "Moral principles should be judged by their practical consequences.",
                "Ethics should be experimental and responsive to changing situations.",
                "Moral progress comes through testing ideas in experience."
            ]
        }
    
    def init_counterfactual_variations(self) -> Dict[str, List[Dict]]:
        """
        Initialize counterfactual variations for scenario testing.
        """
        return {
            'trolley_problem': [
                {'variation': 'standard', 'description': 'Divert trolley to kill one instead of five'},
                {'variation': 'footbridge', 'description': 'Push person off bridge to stop trolley'},
                {'variation': 'loop', 'description': 'Trolley loops back to kill five unless one person stops it'},
                {'variation': 'fat_villain', 'description': 'Person on track is villain who set up scenario'}
            ],
            'privacy_dilemma': [
                {'variation': 'standard', 'description': 'Access private data to prevent harm'},
                {'variation': 'consent', 'description': 'Same scenario but with prior general consent'},
                {'variation': 'public_figure', 'description': 'Same scenario but target is public figure'},
                {'variation': 'severity', 'description': 'Same scenario but with more severe potential harm'}
            ],
            'resource_allocation': [
                {'variation': 'standard', 'description': 'Distribute limited resources among population'},
                {'variation': 'triage', 'description': 'Same scenario but in emergency medical context'},
                {'variation': 'family', 'description': 'Same scenario but recipients include family members'},
                {'variation': 'merit', 'description': 'Same scenario but with information about recipient merit'}
            ]
        }
    
    def init_axiom_tests(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize axiomatic tests for ethical frameworks.
        """
        return {
            'utilitarianism': {
                'greater_good': 'It is justified to harm one person to save many others',
                'hedonism': 'Pleasure is the only intrinsic good and pain the only intrinsic bad',
                'consequentialism': 'The rightness of actions depends solely on their consequences',
                'welfare_equality': 'Each person\'s welfare matters equally'
            },
            'deontology': {
                'categorical_imperative': 'Act only according to that maxim by which you can at the same time will that it should become a universal law',
                'ends_in_themselves': 'Treat humanity always as an end and never merely as a means',
                'perfect_duties': 'Some duties must never be violated regardless of consequences',
                'moral_worth': 'The moral worth of an action lies in the intention behind it'
            },
            'virtue_ethics': {
                'golden_mean': 'Virtue lies in the middle between excess and deficiency',
                'eudaimonia': 'The goal of ethics is to achieve human flourishing',
                'character': 'Character traits are more fundamental than rules or consequences',
                'practical_wisdom': 'Moral judgment requires practical wisdom gained through experience'
            },
            'care_ethics': {
                'relationality': 'Morality is fundamentally about maintaining caring relationships',
                'particularity': 'Moral judgments should respond to particular situations rather than abstract principles',
                'emotion': 'Emotions like empathy and care are essential to moral judgment',
                'contextualism': 'Moral judgments must be sensitive to context and relationships'
            }
        }

    def flatten_nested_dict(self, nested_dict: Dict, prefix='') -> Dict:
        """
        Flatten a nested dictionary structure into a single-level dictionary.
        """
        flattened = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self.flatten_nested_dict(value, new_key))
            else:
                flattened[new_key] = value
        return flattened
    
    def get_all_ethical_terms(self) -> Dict[str, List[str]]:
        """
        Return a dict mapping top-level principle names to a list of all terms
        (including sub-principles).
        """
        all_terms = {}
        for principle, content in self.ethical_principles.items():
            terms = content['terms'].copy()
            for sub_principle, sub_terms in content.get('sub_principles', {}).items():
                terms.extend(sub_terms)
            all_terms[principle] = terms
        return all_terms

    def analyze_dictionary_matches(self, text: str) -> Dict[str, Dict[str, Union[float, Dict]]]:
        """
        Analyze text using the hierarchical dictionary approach.
        
        Returns:
            Nested dictionary of scores for ethical principles and sub-principles
        """
        text_lower = text.lower()
        doc = self.nlp(text)
        sentences = [sent.text.lower() for sent in doc.sents]
        
        results = {}
        
        # Helper function to calculate weighted count
        def get_weighted_count(term_list, context_weight=1.5):
            term_counts = {}
            for term in term_list:
                count = text_lower.count(term.lower())
                term_sentences = [s for s in sentences if term.lower() in s]
                avg_sentence_length = np.mean([len(s.split()) for s in term_sentences]) if term_sentences else 0
                # Simple weighting: longer sentences might indicate more elaboration
                term_weight = 1.0 + (context_weight * (avg_sentence_length / 20.0)) if avg_sentence_length else 1.0
                term_counts[term] = count * term_weight
            return term_counts, sum(term_counts.values())
        
        # Process ethical principles
        for principle, content in self.ethical_principles.items():
            principle_terms, principle_score = get_weighted_count(content['terms'])
            
            sub_results = {}
            for sub_principle, sub_terms in content.get('sub_principles', {}).items():
                sub_term_counts, sub_score = get_weighted_count(sub_terms)
                if principle_score + sub_score > 0:
                    sub_results[sub_principle] = {
                        'score': sub_score,
                        'percentage': sub_score / (principle_score + sub_score),
                        'term_counts': sub_term_counts
                    }
            
            results[principle] = {
                'score': principle_score,
                'term_counts': principle_terms,
                'sub_principles': sub_results
            }
        
        # Normalize
        total_score = sum(data['score'] for data in results.values())
        if total_score > 0:
            for principle in results:
                results[principle]['normalized_score'] = results[principle]['score'] / total_score
        
        return results

    def analyze_semantic_similarity(self, text: str) -> Dict[str, float]:
        """
        Analyze semantic similarity between the text and canonical ethical framework statements.
        """
        if not self.use_embeddings:
            return {}
        
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        try:
            text_embeddings = self.sentence_model.encode(sentences)
            similarities = {}
            
            for framework, statements in self.ethical_framework_statements.items():
                statement_embeddings = self.sentence_model.encode(statements)
                framework_sims = []
                
                for text_emb in text_embeddings:
                    # Max similarity to any statement in the framework
                    statement_sims = [cosine_similarity([text_emb], [stmt_emb])[0][0] 
                                      for stmt_emb in statement_embeddings]
                    framework_sims.append(max(statement_sims) if statement_sims else 0.0)
                
                # Average top 3
                if framework_sims:
                    top_n = min(3, len(framework_sims))
                    similarities[framework] = np.mean(sorted(framework_sims, reverse=True)[:top_n])
                else:
                    similarities[framework] = 0.0
            
            total = sum(similarities.values())
            if total > 0:
                for k in similarities:
                    similarities[k] = similarities[k] / total
            
            return similarities
        except Exception as e:
            print(f"Error in semantic similarity analysis: {e}")
            return {}

    def extract_moral_reasoning_paths(self, text: str) -> Dict[str, Union[Dict, List]]:
        """
        Extract moral reasoning paths from text using a simple graph-based approach.
        """
        doc = self.nlp(text)
        
        # Reasoning connectors
        reasoning_connectors = [
            'because', 'since', 'therefore', 'thus', 'hence', 'so', 'consequently',
            'as a result', 'which means', 'leads to', 'implies', 'suggests',
            'for this reason', 'it follows that', 'given that', 'due to'
        ]
        
        reasoning_sentences = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(conn in sent_text for conn in reasoning_connectors):
                reasoning_sentences.append(sent.text)
        
        G = nx.DiGraph()
        moral_terms = set()
        
        # Extract moral terms present in text
        for principle_dict in self.ethical_principles.values():
            for term in principle_dict['terms']:
                if term.lower() in text.lower():
                    moral_terms.add(term)
            for sub_terms in principle_dict.get('sub_principles', {}).values():
                for st in sub_terms:
                    if st.lower() in text.lower():
                        moral_terms.add(st)
        
        # Identify reasoning paths
        for i, sent in enumerate(reasoning_sentences):
            G.add_node(i, text=sent)
            if i < len(reasoning_sentences) - 1:
                G.add_edge(i, i+1, type='sequence')
            
            # Check for moral terms in this sentence
            sentence_terms = [m for m in moral_terms if m.lower() in sent.lower()]
            if sentence_terms:
                G.nodes[i]['terms'] = G.nodes[i].get('terms', []) + sentence_terms
        
        path_metrics = {
            'num_reasoning_steps': len(reasoning_sentences),
            'moral_terms_in_reasoning': len({term for n in G.nodes for term in G.nodes[n].get('terms', [])}),
            'longest_path_length': 0,
            'reasoning_density': len(reasoning_sentences) / len(list(doc.sents)) if doc.sents else 0
        }
        
        # Find longest path
        longest_path = 0
        for source in G.nodes:
            for target in G.nodes:
                if source != target:
                    try:
                        path = nx.shortest_path(G, source=source, target=target)
                        longest_path = max(longest_path, len(path))
                    except nx.NetworkXNoPath:
                        pass
        path_metrics['longest_path_length'] = longest_path
        
        # Extract reasoning chains
        reasoning_chains = []
        if reasoning_sentences:
            current_chain = []
            for i, sent in enumerate(reasoning_sentences):
                current_chain.append(sent)
                if i < len(reasoning_sentences) - 1 and not G.has_edge(i, i+1):
                    if len(current_chain) > 1:
                        reasoning_chains.append(current_chain)
                    current_chain = []
            if len(current_chain) > 1:
                reasoning_chains.append(current_chain)
        
        return {
            'metrics': path_metrics,
            'reasoning_chains': reasoning_chains,
            'moral_terms': list(moral_terms)
        }

    def analyze_counterfactual_alignment(self, framework: str, responses: Dict[str, str]) -> Dict[str, Union[float, Dict]]:
        """
        Analyze how consistent an ethical framework is across various counterfactual scenario responses.
        """
        if not self.use_embeddings:
            return {'error': 'Embeddings not available for counterfactual analysis'}
        
        framework_statements = self.ethical_framework_statements.get(framework, [])
        if not framework_statements:
            return {'error': f'Framework {framework} not found'}
        
        framework_embeddings = self.sentence_model.encode(framework_statements)
        
        variation_scores = {}
        for variation, response in responses.items():
            doc = self.nlp(response)
            sentences = [sent.text for sent in doc.sents]
            
            if not sentences:
                variation_scores[variation] = 0.0
                continue
            
            response_embeddings = self.sentence_model.encode(sentences)
            max_sims = []
            for r_emb in response_embeddings:
                sims = [cosine_similarity([r_emb], [fw_emb])[0][0] for fw_emb in framework_embeddings]
                max_sims.append(max(sims) if sims else 0.0)
            
            top_n = min(3, len(max_sims))
            variation_scores[variation] = float(np.mean(sorted(max_sims, reverse=True)[:top_n]))
        
        if variation_scores:
            consistency = 1.0 - np.std(list(variation_scores.values()))
            overall = float(np.mean(list(variation_scores.values())))
        else:
            consistency = 0.0
            overall = 0.0
        
        return {
            'variation_scores': variation_scores,
            'consistency': consistency,
            'overall_alignment': overall
        }

    def analyze_axiom_consistency(self, responses: Dict[str, str]) -> Dict[str, Dict[str, Union[float, Dict]]]:
        """
        Analyze how consistently an ethical system's axioms are applied.
        """
        results = {}
        for framework, axioms in self.axiom_tests.items():
            framework_results = {}
            for axiom_name, axiom_statement in axioms.items():
                if axiom_name in responses:
                    matches = self.analyze_dictionary_matches(responses[axiom_name])
                    fw_score = matches.get(framework, {}).get('normalized_score', 0.0)
                    
                    if self.use_embeddings:
                        similarity = self.analyze_semantic_similarity(responses[axiom_name])
                        sem_score = similarity.get(framework, 0.0)
                        fw_score = 0.5 * fw_score + 0.5 * sem_score
                    framework_results[axiom_name] = fw_score
            
            if framework_results:
                consistency = 1.0 - np.std(list(framework_results.values()))
                overall = float(np.mean(list(framework_results.values())))
                results[framework] = {
                    'axiom_scores': framework_results,
                    'consistency': consistency,
                    'overall_alignment': overall
                }
        return results

    def analyze_ethical_profile(self, text: str) -> Dict[str, Union[Dict, List]]:
        """
        Perform a comprehensive ethical analysis of a text.
        """
        results = {}
        
        results['dictionary_analysis'] = self.analyze_dictionary_matches(text)
        if self.use_embeddings:
            results['semantic_analysis'] = self.analyze_semantic_similarity(text)
        else:
            results['semantic_analysis'] = {}
        
        # Moral reasoning paths
        results['reasoning_analysis'] = self.extract_moral_reasoning_paths(text)
        
        # Identify top 3 principles by normalized score (dictionary analysis)
        if 'dictionary_analysis' in results:
            top_principles = sorted(
                [
                    (principle, data.get('normalized_score', 0)) 
                    for principle, data in results['dictionary_analysis'].items()
                ],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            results['top_principles'] = {
                'principles': [p[0] for p in top_principles if p[1] > 0],
                'scores': [p[1] for p in top_principles if p[1] > 0]
            }
        
        # Safety considerations analysis
        safety_results = {}
        for category, content in self.safety_categories.items():
            cat_score = 0
            text_lower = text.lower()
            for term in content['terms']:
                cat_score += text_lower.count(term.lower())
            
            sub_results = {}
            for sub_cat, sub_terms in content.get('sub_categories', {}).items():
                sc_score = 0
                for st in sub_terms:
                    sc_score += text_lower.count(st.lower())
                if sc_score > 0:
                    sub_results[sub_cat] = sc_score
            
            if cat_score > 0 or sub_results:
                safety_results[category] = {
                    'score': cat_score,
                    'sub_categories': sub_results
                }
        
        total_safety = sum(d['score'] for d in safety_results.values()) if safety_results else 0
        if total_safety > 0:
            for cat in safety_results:
                safety_results[cat]['normalized_score'] = safety_results[cat]['score'] / total_safety
        
        results['safety_analysis'] = safety_results
        
        return results

    def compare_responses(self, responses: Dict[str, str]) -> Dict[str, Union[Dict, np.ndarray]]:
        """
        Compare ethical profiles across multiple responses.
        """
        profiles = {resp_id: self.analyze_ethical_profile(text) for resp_id, text in responses.items()}
        
        comparison = {
            'principle_dominance': {},
            'reasoning_complexity': {},
            'safety_emphasis': {},
            'similarity_matrix': {}
        }
        
        # Compare principle dominance
        for principle in self.ethical_principle_keys:
            principle_scores = {}
            for resp_id, profile in profiles.items():
                dict_analysis = profile.get('dictionary_analysis', {})
                principle_data = dict_analysis.get(principle, {})
                principle_scores[resp_id] = principle_data.get('normalized_score', 0)
            
            if any(score > 0 for score in principle_scores.values()):
                comparison['principle_dominance'][principle] = principle_scores
        
        # Compare reasoning complexity
        for resp_id, profile in profiles.items():
            reasoning_data = profile.get('reasoning_analysis', {}).get('metrics', {})
            comparison['reasoning_complexity'][resp_id] = {
                'steps': reasoning_data.get('num_reasoning_steps', 0),
                'density': reasoning_data.get('reasoning_density', 0),
                'moral_terms': reasoning_data.get('moral_terms_in_reasoning', 0)
            }
        
        # Compare safety emphasis
        for category in self.safety_category_keys:
            category_scores = {}
            for resp_id, profile in profiles.items():
                cat_data = profile.get('safety_analysis', {}).get(category, {})
                category_scores[resp_id] = cat_data.get('normalized_score', 0)
            if any(score > 0 for score in category_scores.values()):
                comparison['safety_emphasis'][category] = category_scores
        
        # Similarity matrix if embeddings available
        if self.use_embeddings:
            resp_ids = list(responses.keys())
            n = len(resp_ids)
            matrix = np.zeros((n, n))
            for i in range(n):
                text_i = responses[resp_ids[i]]
                emb_i = self.sentence_model.encode(text_i)
                mean_i = np.mean(emb_i, axis=0).reshape(1, -1)
                for j in range(i, n):
                    text_j = responses[resp_ids[j]]
                    emb_j = self.sentence_model.encode(text_j)
                    mean_j = np.mean(emb_j, axis=0).reshape(1, -1)
                    sim_ij = cosine_similarity(mean_i, mean_j)[0][0]
                    matrix[i, j] = sim_ij
                    matrix[j, i] = sim_ij
            comparison['similarity_matrix'] = {
                'matrix': matrix.tolist(),
                'response_ids': resp_ids
            }
        
        return comparison

    def generate_ethical_report(self, responses: Dict[str, str], scenario_type: str = None) -> Dict:
        """
        Generate a comprehensive ethical analysis report for a set of responses.
        """
        report = {
            'individual_profiles': {},
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Build individual profiles
        for model, text in responses.items():
            report['individual_profiles'][model] = self.analyze_ethical_profile(text)
        
        # Add comparative analysis
        report['comparative_analysis'] = self.compare_responses(responses)
        
        # If scenario_type is provided, do counterfactual analysis (if relevant)
        if scenario_type and scenario_type in self.counterfactual_variations:
            # Example placeholder: not fully integrated
            pass
        
        # Determine dominant frameworks
        dominant_frameworks = {}
        for model, profile in report['individual_profiles'].items():
            sem = profile.get('semantic_analysis', {})
            dict_analysis = profile.get('dictionary_analysis', {})
            
            combined_scores = {}
            # Weighted combination (0.6 semantic, 0.4 dictionary) for demonstration
            for fw, fw_score in sem.items():
                combined_scores[fw] = fw_score * 0.6
            
            for principle, data in dict_analysis.items():
                # Simplify principle->framework mapping
                if principle == 'consequentialism':
                    fw_name = 'utilitarianism'
                elif principle == 'deontological':
                    fw_name = 'kantian_deontology'
                else:
                    fw_name = principle  # fallback
                
                existing = combined_scores.get(fw_name, 0.0)
                combined_scores[fw_name] = existing + 0.4 * data.get('normalized_score', 0.0)
            
            if combined_scores:
                top_fw = max(combined_scores.items(), key=lambda x: x[1])
                dominant_frameworks[model] = {
                    'framework': top_fw[0],
                    'score': top_fw[1],
                    'all_scores': combined_scores
                }
        
        report['dominant_frameworks'] = dominant_frameworks
        
        return report

    def visualize_ethical_profiles(self, report: Dict, output_dir: str = './') -> Dict[str, str]:
        """
        Generate visualizations for ethical profiles and save them.
        """
        os.makedirs(output_dir, exist_ok=True)
        visuals = {}
        
        # 1) Ethical framework alignment
        if 'dominant_frameworks' in report:
            models = list(report['dominant_frameworks'].keys())
            all_fws = set()
            for model_data in report['dominant_frameworks'].values():
                all_fws.update(model_data.get('all_scores', {}).keys())
            all_fws = list(all_fws)
            
            fw_scores = {fw: [] for fw in all_fws}
            for m in models:
                scores = report['dominant_frameworks'][m]['all_scores']
                for fw in all_fws:
                    fw_scores[fw].append(scores.get(fw, 0.0))
            
            x = np.arange(len(models))
            width = 0.8 / len(all_fws)
            fig, ax = plt.subplots(figsize=(12, 7))
            for i, fw in enumerate(all_fws):
                ax.bar(x + i*width - 0.4 + width/2, fw_scores[fw], width, label=fw.title())
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel("Framework Alignment Score")
            ax.set_title("Ethical Framework Alignment by Model")
            ax.legend()
            plt.tight_layout()
            framework_file = os.path.join(output_dir, 'framework_alignment.png')
            plt.savefig(framework_file)
            plt.close()
            visuals['framework_alignment'] = framework_file
        
        # 2) Reasoning complexity
        if 'comparative_analysis' in report and 'reasoning_complexity' in report['comparative_analysis']:
            reasoning_data = report['comparative_analysis']['reasoning_complexity']
            models = list(reasoning_data.keys())
            metrics = ['steps', 'density', 'moral_terms']
            metric_labels = ['Reasoning Steps', 'Reasoning Density', 'Moral Terms']
            
            x = np.arange(len(models))
            width = 0.8 / len(metrics)
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for i, metric in enumerate(metrics):
                vals = []
                for model in models:
                    vals.append(reasoning_data[model].get(metric, 0.0))
                ax.bar(x + i*width - 0.4 + width/2, vals, width, label=metric_labels[i])
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel("Reasoning Metric Value")
            ax.set_title("Reasoning Complexity Comparison")
            ax.legend()
            plt.tight_layout()
            reasoning_file = os.path.join(output_dir, 'reasoning_complexity.png')
            plt.savefig(reasoning_file)
            plt.close()
            visuals['reasoning_complexity'] = reasoning_file
        
        # 3) Safety emphasis
        if ('comparative_analysis' in report 
            and 'safety_emphasis' in report['comparative_analysis']
            and report['comparative_analysis']['safety_emphasis']):
            safety_data = report['comparative_analysis']['safety_emphasis']
            categories = list(safety_data.keys())
            models = list(next(iter(safety_data.values())).keys())
            
            x = np.arange(len(models))
            width = 0.8 / len(categories)
            fig, ax = plt.subplots(figsize=(12, 7))
            
            for i, cat in enumerate(categories):
                cat_vals = [safety_data[cat].get(m, 0.0) for m in models]
                ax.bar(x + i*width - 0.4 + width/2, cat_vals, width, label=cat.title())
            
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel("Safety Emphasis")
            ax.set_title("Safety Considerations Comparison")
            ax.legend()
            plt.tight_layout()
            safety_file = os.path.join(output_dir, 'safety_comparison.png')
            plt.savefig(safety_file)
            plt.close()
            visuals['safety_comparison'] = safety_file
        
        # 4) Similarity heatmap
        if ('comparative_analysis' in report 
            and 'similarity_matrix' in report['comparative_analysis']
            and 'matrix' in report['comparative_analysis']['similarity_matrix']):
            sim_data = report['comparative_analysis']['similarity_matrix']
            matrix = np.array(sim_data['matrix'])
            resp_ids = sim_data['response_ids']
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='YlGnBu', 
                        xticklabels=resp_ids, yticklabels=resp_ids, ax=ax)
            ax.set_title("Response Similarity Heatmap")
            plt.tight_layout()
            sim_file = os.path.join(output_dir, 'similarity_heatmap.png')
            plt.savefig(sim_file)
            plt.close()
            visuals['similarity_heatmap'] = sim_file
        
        return visuals

    def analyze_scenarios(self, output_dir: str = './reports') -> Dict[str, Dict]:
        """
        Analyze all scenarios in the dataset and generate reports.
        """
        results = {}
        os.makedirs(output_dir, exist_ok=True)
        
        for scenario in self.scenarios:
            # We assume each scenario has a unique 'id' or fallback to scenario name
            scenario_id = scenario.get('id', scenario.get('scenario', 'unknown')).replace(' ', '_')
            scenario_type = scenario.get('type', None)
            # 'responses' might be a dict mapping model->text, or a list.
            # Make sure it’s a dict in your JSON structure.
            responses = scenario.get('responses', {})
            if not responses:
                continue
            
            report = self.generate_ethical_report(responses, scenario_type)
            scenario_dir = os.path.join(output_dir, f'scenario_{scenario_id}')
            os.makedirs(scenario_dir, exist_ok=True)
            
            visuals = self.visualize_ethical_profiles(report, scenario_dir)
            report['visualizations'] = visuals
            
            report_file = os.path.join(scenario_dir, 'report.json')
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            results[scenario_id] = {
                'report': report,
                'report_file': report_file
            }
        
        self.generate_summary_report(results, output_dir)
        return results

    def generate_summary_report(self, scenario_results: Dict[str, Dict], output_dir: str) -> str:
        """
        Generate a summary report across all scenarios and save it as JSON or text.
        """
        summary = {
            'model_framework_alignment': {},
            'model_reasoning_complexity': {},
            'model_safety_emphasis': {},
            'scenario_count': len(scenario_results),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        all_models = set()
        for _, data in scenario_results.items():
            rpt = data['report']
            all_models.update(rpt.get('dominant_frameworks', {}).keys())
        
        all_models = list(all_models)
        # Initialize
        for m in all_models:
            summary['model_framework_alignment'][m] = {
                'framework_counts': {},
                'average_scores': {}
            }
            summary['model_reasoning_complexity'][m] = {
                'avg_steps': 0.0,
                'avg_density': 0.0,
                'avg_moral_terms': 0.0
            }
            summary['model_safety_emphasis'][m] = {cat: 0.0 for cat in self.safety_category_keys}
        
        scenario_count_map = {m: 0 for m in all_models}
        
        for _, data in scenario_results.items():
            rpt = data['report']
            # Framework alignment
            for model, fw_data in rpt.get('dominant_frameworks', {}).items():
                if model not in all_models:
                    continue
                scenario_count_map[model] += 1
                fw_name = fw_data.get('framework')
                if fw_name:
                    summary['model_framework_alignment'][model]['framework_counts'][fw_name] = \
                        summary['model_framework_alignment'][model]['framework_counts'].get(fw_name, 0) + 1
                
                for fw, score in fw_data.get('all_scores', {}).items():
                    if fw not in summary['model_framework_alignment'][model]['average_scores']:
                        summary['model_framework_alignment'][model]['average_scores'][fw] = []
                    summary['model_framework_alignment'][model]['average_scores'][fw].append(score)
            
            # Reasoning complexity
            rc_data = rpt.get('comparative_analysis', {}).get('reasoning_complexity', {})
            for model, metrics in rc_data.items():
                if model not in all_models:
                    continue
                summary['model_reasoning_complexity'][model]['avg_steps'] += metrics.get('steps', 0)
                summary['model_reasoning_complexity'][model]['avg_density'] += metrics.get('density', 0)
                summary['model_reasoning_complexity'][model]['avg_moral_terms'] += metrics.get('moral_terms', 0)
            
            # Safety emphasis
            s_emphasis = rpt.get('comparative_analysis', {}).get('safety_emphasis', {})
            for cat, cat_data in s_emphasis.items():
                for model, score in cat_data.items():
                    if model in all_models:
                        summary['model_safety_emphasis'][model][cat] += score
        
        # Compute averages
        for m in all_models:
            c = scenario_count_map[m]
            if c == 0:
                continue
            # average framework scores
            for fw, score_list in summary['model_framework_alignment'][m]['average_scores'].items():
                if score_list:
                    summary['model_framework_alignment'][m]['average_scores'][fw] = \
                        float(np.mean(score_list))
                else:
                    summary['model_framework_alignment'][m]['average_scores'][fw] = 0.0
            
            # reasoning complexity
            summary['model_reasoning_complexity'][m]['avg_steps'] /= c
            summary['model_reasoning_complexity'][m]['avg_density'] /= c
            summary['model_reasoning_complexity'][m]['avg_moral_terms'] /= c
            
            # safety emphasis
            for cat in self.safety_category_keys:
                summary['model_safety_emphasis'][m][cat] /= c
        
        summary_path = os.path.join(output_dir, 'summary_report.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path

# Example usage entrypoint
def main():
    data_file = "multi_model_responses.json"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} does not exist. Please generate responses first.")
        return
    
    analyzer = EnhancedEthicsAnalyzer(data_file)
    results = analyzer.analyze_scenarios(output_dir="./reports")
    print(f"Analysis complete! Reports are stored under './reports'.")

if __name__ == "__main__":
    main()
