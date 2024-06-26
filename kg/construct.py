import os
from transition_amr_parser.parse import AMRParser
import json
import networkx as nx
import penman
from penman import transform as penman_transform
from penman.models import amr as penman_amr
from fastcoref import FCoref, LingMessCoref
import itertools
from contextlib import contextmanager
import copy
import torch
from loguru import logger
import traceback
import sys

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import logging
logging.getLogger('penman').setLevel(logging.ERROR)


class KGCreator:
    def __init__(self, BATCH_SIZE=64, parser=None):
        self.BATCH_SIZE = BATCH_SIZE
        self.SENTENCE_COUNT = 0
        self.parser = parser

    @staticmethod
    def get_token_map(original_graph, modified_graph):
        mapping = {}
        for i, triple in enumerate(original_graph.triples):
            mapping[modified_graph.triples[i][0]] = triple[0]
        return mapping

    def process_sentences(self, sentence_list, multiDocKG, cluster_id, doc_id):
        batch = []

        existSentenceCount = len(multiDocKG.metadata[doc_id]['sentences'].keys())
        tokens = []

        for sentence_index, sentence in enumerate(sentence_list):
            sentence_tokens, _ = self.parser.tokenize(sentence)

            batch.append(sentence_tokens)
            if len(batch) == self.BATCH_SIZE or sentence_index == len(sentence_list) - 1:
                annotations_list, decoding_data_list = self.parser.parse_sentences(
                    batch)
                batch = []
                for batch_snt_idx, (annotation, decoding_data) in enumerate(zip(annotations_list, decoding_data_list)):
                    actual_sentence_index = sentence_index - \
                        len(annotations_list) + 1 + \
                        batch_snt_idx + existSentenceCount

                    ibm_amr_graph = decoding_data.get_amr()
                    
                    tokens.extend(ibm_amr_graph.tokens)
                    
                    sentence_amr_penman = ibm_amr_graph.to_penman(isi=False)
                    tree = penman.parse(sentence_amr_penman)
                    tree.reset_variables('{}-{}-{}-'.format(cluster_id, doc_id, actual_sentence_index) + 'z{i}')
                    new_ibm_amr_graph = penman.interpret(tree)
                    multiDocKG.metadata[doc_id]['sentences'][actual_sentence_index] = {
                        # a list of tokens tokenized from the sentence
                        'tokens': ibm_amr_graph.tokens,
                        # a mapping with the key being the token(variable in penman) in the AMR graph from the IBM parser \
                        # and the value being the mapped word in the sentence or the word generated by the IBM parser
                        'nodes': ibm_amr_graph.nodes,
                        # A mapping with the key being the token(variable in penman) in the AMR graph from the IBM parser \
                        # and the value being the index of the token in the sentence
                        'alignments': ibm_amr_graph.alignments,
                        # Get a mapping with the key being the token(variable in penman) in the AMR graph \
                        # and the value being the token in the AMR graph from the IBM parser
                        'token_map': self.get_token_map(penman.decode(sentence_amr_penman), new_ibm_amr_graph),
                    }
                    self.handle_multiple_sentences(cluster_id, doc_id, actual_sentence_index, new_ibm_amr_graph)
                    multiDocKG.triples.extend(new_ibm_amr_graph.triples)
                    multiKGtree = penman.configure(multiDocKG)
                    # multiKGtree = penman_transform.canonicalize_roles(
                    #     multiKGtree, model=penman_amr.model
                    # )
                    penman.layout.rearrange(
                        multiKGtree, key=penman_amr.model.canonical_order)
                    multiDocKG = penman.interpret(multiKGtree)
                    
        
                    
        
        return multiDocKG, tokens, new_ibm_amr_graph

    def handle_multiple_sentences(self, cluster_id, doc_id, actual_sentence_index, new_ibm_amr_graph):
        if ('{}-{}-{}-z0'.format(cluster_id, doc_id, actual_sentence_index), ':instance', 'multi-sentence') in new_ibm_amr_graph.triples:
            pop_index = new_ibm_amr_graph.triples.index(
                ('{}-{}-{}-z0'.format(cluster_id, doc_id, actual_sentence_index), ':instance', 'multi-sentence'))
            pop_source = new_ibm_amr_graph.triples[pop_index][0]
            new_ibm_amr_graph.triples.pop(pop_index)
            for triple_idx, triple in enumerate(new_ibm_amr_graph.triples):
                if triple[0] == pop_source:
                    original_source, original_role, original_target = triple
                    if original_role.startswith(":snt"):
                        new_ibm_amr_graph.triples[triple_idx] = (
                            "origin", ':snt{}'.format(self.SENTENCE_COUNT), original_target)
                        self.SENTENCE_COUNT += 1
                    else:
                        new_ibm_amr_graph.triples[triple_idx] = (
                            "origin", original_role, original_target)
        else:
            new_ibm_amr_graph.triples.append(
                ('origin', ':snt{}'.format(self.SENTENCE_COUNT), new_ibm_amr_graph.top))
            self.SENTENCE_COUNT += 1

    def create_maintext(self, multiDocKG, doc_id):
        new_maintext = str()
        for _, value in multiDocKG.metadata[doc_id]['sentences'].items():
            sentence = " ".join(value['tokens'])
            new_maintext = (new_maintext + " " + sentence).strip()

        multiDocKG.metadata[doc_id]['maintext'] = new_maintext
        
        return multiDocKG
    
    def createKGFromSentence(self, sentence_list, multiDocKG, cluster_id, doc_id, modal='text'):
        
        multiDocKG, tokens, penman_graph = self.process_sentences(sentence_list, multiDocKG, cluster_id, doc_id)
        
        if modal == 'text':
            for triple in penman_graph.triples:
                multiDocKG.metadata[doc_id]['text_triples'].append(triple)
            multiDocKG.metadata[doc_id]['text'] = ' '.join(tokens)    
        elif modal == 'image':
            for triple in penman_graph.triples:
                multiDocKG.metadata[doc_id]['image_triples'].append(triple)
            multiDocKG.metadata[doc_id]['image_caption'] = ' '.join(tokens)
        elif modal == 'video':
            for triple in penman_graph.triples:
                multiDocKG.metadata[doc_id]['video_triples'].append(triple)
            multiDocKG.metadata[doc_id]['video_caption'] = ' '.join(tokens)
        elif modal == 'audio':
            for triple in penman_graph.triples:
                multiDocKG.metadata[doc_id]['audio_triples'].append(triple)
            multiDocKG.metadata[doc_id]['audio_caption'] = ' '.join(tokens)
        else:
            raise ValueError("Invalid modal")
        
        return self.create_maintext(multiDocKG, doc_id)


class EntityCoreference:
    def __init__(self):
        self.coref_model = FCoref(device='cpu')

    @contextmanager
    def process_metadata(self, kg, doc_idx, sent_idx):
        meta = kg.metadata[doc_idx]['sentences'][sent_idx]
        token_map = meta['token_map']
        yield meta, token_map

    def adjust_indices(self, index, offset):
        return index - offset

    def get_nodes(self, meta, token_map, mention_start, mention_end):
        alignment_token_idx = [
            k for k, v in meta['alignments'].items() if v[0] in range(mention_start, mention_end)
        ]
        return [
            key for idx in alignment_token_idx for key, val in token_map.items() if idx == val
        ]

    def get_sentence_index(self, kg, doc_idx, token_index):
        accumulated_length = 0
        for snt_idx, value in kg.metadata[doc_idx]['sentences'].items():
            accumulated_length += len(value['tokens'])
            if accumulated_length > token_index:
                return snt_idx, accumulated_length - len(value['tokens'])
        raise ValueError(
            f"Token index {token_index} exceeds length of document {doc_idx}")

    def get_doc_index(self, kg, token_index):
        accumulated_length = 0
        for doc_id in kg.metadata['doc_ids']:
            tokens = kg.metadata[doc_id]['maintext'].split(" ")
            accumulated_length += len(tokens)
            if accumulated_length > token_index:
                return doc_id, accumulated_length - len(tokens)
        raise ValueError(
            f"Word index {token_index} exceeds length of entire text")

    def resolve(self, kg):
        KG_TEXT = ' '.join(kg.metadata[d]['maintext'] for d in kg.metadata['doc_ids']).strip()
        
        logger.info(KG_TEXT)
        
        # for doc_id in kg.metadata['doc_ids']:
        #     logger.info('doc id: {}', doc_id)
        #     logger.info('maintext: {}', kg.metadata[doc_id]['maintext'])
        
        tokens = KG_TEXT.split(" ")
        # TODO: batchtize: max_tokens_in_batch
        predictions = self.coref_model.predict(
            texts=[tokens], is_split_into_words=True, max_tokens_in_batch=100)[0]
        
        # TODO: remove indexing
        clusters = predictions.get_clusters(as_strings=False)
        # logger.info('Found {} clusters', len(clusters))
        # logger.info('cluters: {}', predictions.get_clusters()[-9])
        # logger.info('cluters: {}', predictions.get_clusters(as_strings=False)[-9])

        for cluster in clusters:
            # logger.info("cluster in loop: {}", cluster)
            for mention1_idx, mention2_idx in itertools.combinations(range(len(cluster)), 2):
                # logger.info("mention1: {}, results: {}", mention1_idx, cluster[mention1_idx])
                # logger.info("mention2: {}, results: {}", mention2_idx, cluster[mention2_idx])
                mention1_start, mention1_end = cluster[mention1_idx]
                mention2_start, mention2_end = cluster[mention2_idx]

                doc1_idx, doc1_offset = self.get_doc_index(kg, mention1_start)
                doc2_idx, doc2_offset = self.get_doc_index(kg, mention2_start)

                mention1_start, mention1_end = map(
                    self.adjust_indices, (mention1_start, mention1_end), (doc1_offset, doc1_offset))
                mention2_start, mention2_end = map(
                    self.adjust_indices, (mention2_start, mention2_end), (doc2_offset, doc2_offset))

                sent1_idx, sent1_offset = self.get_sentence_index(
                    kg, doc1_idx, mention1_start)
                sent2_idx, sent2_offset = self.get_sentence_index(
                    kg, doc2_idx, mention2_start)

                mention1_start, mention1_end = map(
                    self.adjust_indices, (mention1_start, mention1_end), (sent1_offset, sent1_offset))
                mention2_start, mention2_end = map(
                    self.adjust_indices, (mention2_start, mention2_end), (sent2_offset, sent2_offset))

                with self.process_metadata(kg, doc1_idx, sent1_idx) as (meta, key_mapping):
                    mention_nodes_1 = self.get_nodes(
                        meta, key_mapping, mention1_start, mention1_end)

                with self.process_metadata(kg, doc2_idx, sent2_idx) as (meta, key_mapping):
                    mention_nodes_2 = self.get_nodes(
                        meta, key_mapping, mention2_start, mention2_end)
            
                # logger.info("mention_nodes1: {}", mention_nodes_1)
                # logger.info("mention_nodes2: {}", mention_nodes_2)
                for m1, m2 in itertools.product(set(mention_nodes_1), set(mention_nodes_2)):
                    if m1 != m2:
                        triple1 = (m1, ":coref", m2)
                        triple2 = (m2, ":coref", m1)
                        if all(triple not in kg.triples for triple in [triple1, triple2]):
                            # TODO: check if there is a triple with same source and target?
                            if any(((triple[0] == m1 and triple[2] == m2) or (triple[0] == m2 and triple[2] == m1)) for triple in kg.triples):
                                continue
                            kg.triples.extend([triple1, triple2])
                            
                            # TODO: add the coreference metadata
                            if 'coreferences' not in kg.metadata:
                                kg.metadata['coreferences'] = dict()
                            
                            if doc1_idx not in kg.metadata['coreferences']:
                                kg.metadata['coreferences'][doc1_idx] = list()
                            if doc2_idx not in kg.metadata['coreferences']:
                                kg.metadata['coreferences'][doc2_idx] = list()
                            
                            if doc2_idx not in kg.metadata['coreferences'][doc1_idx]:
                                kg.metadata['coreferences'][doc1_idx].append(doc2_idx)
                            
                            if doc1_idx not in kg.metadata['coreferences'][doc2_idx]:
                                kg.metadata['coreferences'][doc2_idx].append(doc1_idx)

        return kg


class GraphConverter:
    def __init__(self, graph, no_escape_characters=False):
        self.graph = graph
        self.no_escape_characters = no_escape_characters

    def handle_instances(self, G):
        for instance in self.graph.instances():
            if instance.target == "multi-sentence":
                continue

            doc_id, snt_idx = map(int, instance.source.split("-")[1:3])
            if doc_id not in self.graph.metadata['doc_ids']:
                continue

            self.update_doc_info(G, doc_id)

            group_id = self.assign_group_id(doc_id, instance)

            if group_id:
                snt_metadata = self.graph.metadata[doc_id]['sentences'][snt_idx]
                token_map = snt_metadata['token_map'][instance.source]
                word_token = snt_metadata['tokens'][snt_metadata['alignments'][token_map][0]]

                G.add_node(instance.source, amr_token=instance.target,
                           word_token=word_token, group=group_id, sim_group='{}'.format(group_id.split('-')[0]))
            else:
                logger.info('No group id for instance: {}', instance.source)

    def handle_edges(self, G):
        for edge in self.graph.edges():
            G.add_edge(edge.source, edge.target, edge_info=edge.role)

    def handle_attributes(self, G):
        for attribute in self.graph.attributes():
            if attribute.source == "origin":
                continue

            doc_id, snt_idx = map(int, attribute.source.split("-")[1:3])
            if doc_id not in self.graph.metadata['doc_ids']:
                continue

            group_id = self.assign_group_id(doc_id, attribute)
            if group_id:
                snt_metadata = self.graph.metadata[doc_id]['sentences'][snt_idx]
                token_map = snt_metadata['token_map'][attribute.source]
                word_token = snt_metadata['tokens'][snt_metadata['alignments'][token_map][0]]

                attribute_node_id = f"{attribute.source}-{attribute.role}-{attribute.target}"
                G.add_node(attribute_node_id, amr_token=attribute.target,
                           word_token=word_token, group=group_id, sim_group='{}'.format(group_id.split('-')[0]))
                G.add_edge(attribute.source, attribute_node_id,
                           edge_info=attribute.role)

    def assign_group_id(self, doc_id, instance):
        group_id = None
        if (instance.source, ':instance', instance.target) in self.graph.metadata[doc_id]['video_triples']:
            group_id = "{}-{}".format(doc_id, "video")
        elif (instance.source, ':instance', instance.target) in self.graph.metadata[doc_id]['audio_triples']:
            group_id = "{}-{}".format(doc_id, "audio")
        elif (instance.source, ':instance', instance.target) in self.graph.metadata[doc_id]['image_triples']:
            group_id = "{}-{}".format(doc_id, "image")
        else:
            group_id = "{}-{}".format(doc_id, "text")

        return group_id

    def update_doc_info(self, G, doc_id):
        G.graph['doc_info'].setdefault(doc_id, {
            'text': self.graph.metadata[doc_id]['text'] if 'text' in self.graph.metadata[doc_id] else '',
            'image': self.graph.metadata[doc_id]['image_caption'] if 'image_caption' in self.graph.metadata[doc_id] else '',
            'video': self.graph.metadata[doc_id]['video_caption'] if 'video_caption' in self.graph.metadata[doc_id] else '',
            'audio': self.graph.metadata[doc_id]['audio_caption'] if 'audio_caption' in self.graph.metadata[doc_id] else '',
            # 'video_filename': self.graph.metadata[doc_id]['video_filename'] if 'video_filename' in self.graph.metadata[doc_id] else '
        })

        G.graph['doc_info'][doc_id]['sentences'] = list()
        for _, val in self.graph.metadata[doc_id]['sentences'].items():
            G.graph['doc_info'][doc_id]['sentences'].append(
                ' '.join(val['tokens']))

    def convert(self):
        G = nx.Graph()
        G.graph['kg_penman'] = self.graph.metadata['kg_penman']
        G.graph['doc_info'] = {}

        G.add_node("origin", amr_token="multi_sentence",
                   word_token="multi-sentence", group=-1, sim_group=-1)
                   
        
        # TODO: coreference metadata
        G.graph['coreferences'] = self.graph.metadata['coreferences']
        
        
        # TODO: add image nodes
        cluster_id = self.graph.metadata['cluster_id']
        for doc_id in self.graph.metadata['doc_ids']:
            G.add_node(
                "{}-{}-image".format(cluster_id, doc_id),
                amr_token="{}-{}-image".format(cluster_id, doc_id),
                word_token="{}-{}-image".format(cluster_id, doc_id),
                group="{}-image".format(doc_id),
                sim_group="{}".format(doc_id),
                image="frame_{}.png".format(doc_id)
            )
            
        # TODO: add video nodes
        # for doc_id in self.graph.metadata['doc_ids']:
        #     G.add_node(
        #         "{}-{}-video".format(cluster_id, doc_id),
        #         amr_token="{}-{}-video".format(cluster_id, doc_id),
        #         word_token="{}-{}-video".format(cluster_id, doc_id),
        #         group="{}-video".format(doc_id),
        #         sim_group="{}".format(doc_id),
                # video="{}.mp4".format(self.grpah.metadata[doc_id]['video_filename'])
        #     )
            

        if self.no_escape_characters:
            for i, triple in enumerate(self.graph.triples):
                self.graph.triples[i] = (triple[0].replace('"', ''),
                                         triple[1], triple[2].replace('"', ''))

        self.handle_instances(G)
        self.handle_edges(G)
        self.handle_attributes(G)

        return G


class PathProcess():
    def __init__(self):
        super().__init__()

        # self.IMAGE_DATA_PATH = '/storage/projects/chiawei/m3dc/image_caption'
        self.SAVE_PATH = './kg_output'

        # self.kg_creator = KGCreator(BATCH_SIZE=32, parser=AMRParser.from_pretrained('AMR3-structbart-L'))
        # self.entity_coreference = EntityCoreference()
        # self.event_coreference = EventCoref()

    def constructKG(self):
        
        kg_creator = KGCreator(
            BATCH_SIZE=32, parser=AMRParser.from_pretrained('AMR3-structbart-L'))
        
        entity_coreference = EntityCoreference()
        # event_coreference = EventCoref()
        
        # for root, dirs, filenames in os.walk(os.path.join(TEXT_DATA_PATH)):
            
        # logger.info("Start constructing KG in {}", root)
        multiDocKG = penman.graph.Graph()
        multiDocKG.metadata['cluster_id'] = str()
        # create a set to store all the document ids
        multiDocKG.metadata['doc_ids'] = set()
        # make 'multi-sentence' as the root node of the graph
        multiDocKG.triples.append(
            ('origin', ':instance', 'multi-sentence'))

        to_be_removed_ids = set()
        # logger.info('root = {}', root)
        cluster_id = '777'
        multiDocKG.metadata['cluster_id'] = cluster_id
        # cluster_id = int(cluster_id)
        
        frames = json.load(open(os.path.join('../', 'captions.json')))
        for frame in frames:
            doc_id = frame['frame'].split('.')[0].split('_')[1]
            doc_id = int(doc_id)
            logger.info('cluster id: {} doc id: {}',cluster_id, doc_id)
            
            image_caption = sent_tokenize(frame['caption'])
                    
            if doc_id not in multiDocKG.metadata['doc_ids']:
                # Add the document ID to the set
                multiDocKG.metadata['doc_ids'].add(doc_id)
                    
                multiDocKG.metadata[doc_id] = dict()
                multiDocKG.metadata[doc_id]['sentences'] = dict()
                # TODO: to prevent disconnected graphs during entity & event coreference, combine all the sentences from the text and image modalities
                multiDocKG.metadata[doc_id]['maintext'] = str()
                
                # to store the seperate text, image, video, audio content
                multiDocKG.metadata[doc_id]['text'] = str()
                multiDocKG.metadata[doc_id]['text_triples'] = list()
                multiDocKG.metadata[doc_id]['image_caption'] = str()
                multiDocKG.metadata[doc_id]['image_triples'] = list()
                multiDocKG.metadata[doc_id]['video_caption'] = str()
                multiDocKG.metadata[doc_id]['video_triples'] = list()
                multiDocKG.metadata[doc_id]['audio_caption'] = str()
                multiDocKG.metadata[doc_id]['audio_triples'] = list()
                
                multiDocKG.metadata[doc_id]['video_filename'] = str()

            multiDocKG = kg_creator.createKGFromSentence(image_caption, multiDocKG, cluster_id, doc_id, modal='image')
            
            if len(multiDocKG.metadata['doc_ids']) == len(to_be_removed_ids):
                logger.debug("Skip cluster: {}, document or image not exist", cluster_id)
                continue

        for doc_id in to_be_removed_ids:
            if doc_id in multiDocKG.metadata['doc_ids']:
                multiDocKG.metadata['doc_ids'].remove(doc_id)
                multiDocKG.metadata.pop(doc_id, None)

        if len(multiDocKG.metadata['doc_ids']) != 0:
            # entity coreference
            logger.debug("Start processing entity coreference")
            multiDocKG = entity_coreference.resolve(multiDocKG)
            
            # logger.info("Finish processing entity coreference")
            
            # TODO: event coreference too slow
            # logger.debug("Start processing event coreference")
            # multiDocKG = event_coreference.resolve(multiDocKG)

            # TODO: unknown issue with penman, so we need to copy the graph and remove the metadata to encode the graph into penman
            tempKG = copy.deepcopy(multiDocKG)
            tempKG.metadata = {}

            try:
                multiDocKG.metadata['kg_penman'] = penman.encode(tempKG, indent=None, compact=True)
            except Exception as e:
                logger.error(e)
                traceback.print_exception(*sys.exc_info())
                # continue

            G = GraphConverter(multiDocKG).convert()
            d = nx.json_graph.node_link_data(G)

            logger.debug('saving file to: {}', os.path.join(self.SAVE_PATH, cluster_id, 'graph.json'))
            os.makedirs(os.path.join(self.SAVE_PATH), exist_ok=True)
            json.dump(d, open(os.path.join(self.SAVE_PATH, 'graph.json'), 'w'))

            # TODO: post-processing for json
            G = GraphConverter(multiDocKG, no_escape_characters=True).convert()
            d = nx.json_graph.node_link_data(G)
            logger.debug('saving file to: {}', os.path.join(self.SAVE_PATH, cluster_id, 'graph_no_quotes.json'))
            json.dump(d, open(os.path.join(self.SAVE_PATH, 'graph_no_quotes.json'), 'w'))
            
        logger.debug("Finish processing cluster: {}", cluster_id)
    
            
PathProcess().constructKG()

