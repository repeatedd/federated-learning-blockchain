from hashlib import sha256
import json
import time
import pickle
import dill

from flask import Flask, request
import requests


class Block:
    def __init__(self, index, cli_model, fin_model, timestamp, previous_hash, cli, nonce=0):
        self.index = index
        self.cli_model = cli_model
        self.fin_model = fin_model
        self.cli = cli
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce

    def compute_hash(self):
        """
        A function that return the hash of the block contents.
        """
        temp_block = self
        temp = []
        temp.append(self.index)
        temp.append(self.cli)
        temp.append(self.timestamp)
        temp.append(self.previous_hash)
        temp.append(self.nonce)
        if temp_block.cli_model != 0:
            for i in range(len(temp_block.cli_model.layers)):
                temp.append(temp_block.cli_model.layers[i].W.tolist())
                temp.append(temp_block.cli_model.layers[i].b.tolist())

        block_bytes = json.dumps(temp)
        return sha256(block_bytes.encode()).hexdigest()


class Blockchain:
    # difficulty of our PoW algorithm
    difficulty = 1

    def __init__(self):
        self.unconfirmed_transactions = []
        self.chain = []

    def create_genesis_block(self):
        """
        A function to generate genesis block and appends it to
        the chain. The block has index 0, previous_hash as 0, and
        a valid hash.
        """
        genesis_block = Block(0, 0, 0, 0, 0, "0")
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)
    
    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        """
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of latest block
          in the chain match.
        """
        previous_hash = self.last_block.hash
        if previous_hash != block.previous_hash:
            return False
        if not Blockchain.is_valid_proof(block, proof):
            return False
        # print("reached")
        block.hash = proof
        self.chain.append(block)
        return self

    def add_blocks(self, chain_dump):
        """
        Add the blocks coming from server
        after verifying them
        """
        for idx, block_data in enumerate(chain_dump):
            # if idx == 0:
            #     continue  # skip genesis block
            block = Block(block_data.index,
                        block_data.cli_model,
                        block_data.fin_model,
                        block_data.cli,
                        block_data.timestamp,
                        block_data.previous_hash,
                        block_data.nonce)
            proof = block_data.hash
            added = self.add_block(block, proof)
            if not added:
                raise Exception("The chain dump is tampered!!")

    @staticmethod
    def proof_of_work(block):
        """
        Function that tries different values of nonce to get a hash
        that satisfies our difficulty criteria.
        """
        block.nonce = 0

        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * Blockchain.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()

        return computed_hash

    def add_new_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)

    @classmethod
    def is_valid_proof(cls, block, block_hash):
        """
        Check if block_hash is valid hash of block and satisfies
        the difficulty criteria.
        print(block.__dict__)
        """
        print("hash calculated", block.compute_hash())
        # return True

        return (block_hash.startswith('0' * Blockchain.difficulty) and
                block_hash == block.compute_hash())

    @classmethod
    def check_chain_validity(cls, chain):
        result = True
        previous_hash = "0"

        for block in chain:
            block_hash = block.hash
            # remove the hash field to recompute the hash again
            # using `compute_hash` method.
            delattr(block, "hash")

            if not cls.is_valid_proof(block, block_hash) or \
                    previous_hash != block.previous_hash:
                result = False
                break

            block.hash, previous_hash = block_hash, block_hash

        return result

    # def mine(self):
    #     """
    #     This function serves as an interface to add the pending
    #     transactions to the blockchain by adding them to the block
    #     and figuring out Proof Of Work.
    #     """
    #     if not self.unconfirmed_transactions:
    #         return False

    #     last_block = self.last_block

    #     new_block = Block(index=last_block.index + 1,
    #                       transactions=self.unconfirmed_transactions,
    #                       timestamp=time.time(),
    #                       previous_hash=last_block.hash)

    #     proof = self.proof_of_work(new_block)
    #     self.add_block(new_block, proof)

        # self.unconfirmed_transactions = []

        # return True




# app = Flask(__name__)

# the node's copy of blockchain
# blockchain = Blockchain()
# blockchain.create_genesis_block()

# # the address to other participating members of the network
# peers = set()


# endpoint to submit a new transaction. This will be used by
# our application to add new data (posts) to the blockchain
# @app.route('/new_transaction', methods=['POST'])
# def new_transaction():
#     tx_data = request.get_json()
#     required_fields = ["author", "content"]

#     for field in required_fields:
#         if not tx_data.get(field):
#             return "Invalid transaction data", 404

#     tx_data["timestamp"] = time.time()

#     blockchain.add_new_transaction(tx_data)

#     return "Success", 201


# endpoint to return the node's copy of the chain.
# Our application will be using this endpoint to query
# all the posts to display.
# @app.route('/chain', methods=['GET'])
# def get_chain():
#     chain_data = []
#     for block in blockchain.chain:
#         chain_data.append(block.__dict__)
#     return json.dumps({"length": len(chain_data),
#                        "chain": chain_data,
#                        "peers": list(peers)})


# endpoint to request the node to mine the unconfirmed
# transactions (if any). We'll be using it to initiate
# a command to mine from our application itself.
# @app.route('/mine', methods=['GET'])
# def mine_unconfirmed_transactions():
#     result = blockchain.mine()
#     if not result:
#         return "No transactions to mine"
#     else:
#         # Making sure we have the longest chain before announcing to the network
#         chain_length = len(blockchain.chain)
#         consensus()
#         if chain_length == len(blockchain.chain):
#             # announce the recently mined block to the network
#             announce_new_block(blockchain.last_block)
#         return "Block #{} is mined.".format(blockchain.last_block.index)


# # endpoint to add new peers to the network.
# @app.route('/register_node', methods=['POST'])
# def register_new_peers():
#     node_address = request.get_json()["node_address"]
#     if not node_address:
#         return "Invalid data", 400

#     # Add the node to the peer list
#     peers.add(node_address)

#     # Return the consensus blockchain to the newly registered node
#     # so that he can sync
#     return get_chain()


# @app.route('/register_with', methods=['POST'])
# def register_with_existing_node():
#     """
#     Internally calls the `register_node` endpoint to
#     register current node with the node specified in the
#     request, and sync the blockchain as well as peer data.
#     """
#     node_address = request.get_json()["node_address"]
#     if not node_address:
#         return "Invalid data", 400

#     data = {"node_address": request.host_url}
#     headers = {'Content-Type': "application/json"}

#     # Make a request to register with remote node and obtain information
#     response = requests.post(node_address + "/register_node",
#                              data=json.dumps(data), headers=headers)

#     if response.status_code == 200:
#         global blockchain
#         global peers
#         # update chain and the peers
#         chain_dump = response.json()['chain']
#         blockchain = create_chain_from_dump(chain_dump)
#         peers.update(response.json()['peers'])
#         return "Registration successful", 200
#     else:
#         # if something goes wrong, pass it on to the API response
#         return response.content, response.status_code


# def add_blocks(chain_dump):
#     # generated_blockchain = Blockchain()
#     # generated_blockchain.create_genesis_block()
#     for idx, block_data in enumerate(chain_dump):
#         if idx == 0:
#             continue  # skip genesis block
#         block = Block(block_data["index"],
#                       block_data["transactions"],
#                       block_data["timestamp"],
#                       block_data["previous_hash"],
#                       block_data["nonce"])
#         proof = block_data['hash']
#         added = generated_blockchain.add_block(block, proof)
#         if not added:
#             raise Exception("The chain dump is tampered!!")
#     return generated_blockchain


# endpoint to add a block mined by someone else to
# the node's chain. The block is first verified by the node
# and then added to the chain.
# @app.route('/add_blocks', methods=['POST'])
# def verify_and_add_blocks(blocks):
#     block_data = request.get_json()
#     block = Block(block_data["index"],
#                   block_data["transactions"],
#                   block_data["timestamp"],
#                   block_data["previous_hash"],
#                   block_data["nonce"])

#     proof = block_data['hash']
#     added = blockchain.add_block(block, proof)

#     if not added:
#         return "The block was discarded by the node", 400

#     return "Block added to the chain", 201


# # endpoint to query unconfirmed transactions
# @app.route('/pending_tx')
# def get_pending_tx():
#     return json.dumps(blockchain.unconfirmed_transactions)


# def consensus():
#     """
#     Our naive consnsus algorithm. If a longer valid chain is
#     found, our chain is replaced with it.
#     """
#     global blockchain

#     longest_chain = None
#     current_len = len(blockchain.chain)

#     for node in peers:
#         response = requests.get('{}chain'.format(node))
#         length = response.json()['length']
#         chain = response.json()['chain']
#         if length > current_len and blockchain.check_chain_validity(chain):
#             current_len = length
#             longest_chain = chain

#     if longest_chain:
#         blockchain = longest_chain
#         return True

#     return False


# def announce_new_block(block):
#     """
#     A function to announce to the network once a block has been mined.
#     Other blocks can simply verify the proof of work and add it to their
#     respective chains.
#     """
#     for peer in peers:
#         url = "{}add_block".format(peer)
#         headers = {'Content-Type': "application/json"}
#         requests.post(url,
#                       data=json.dumps(block.__dict__, sort_keys=True),
#                       headers=headers)

# # Uncomment this line if you want to specify the port number in the code
# #app.run(debug=True, port=8000)
