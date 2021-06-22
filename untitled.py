Flattening changeable-goal environment for agent DDPG
AGENT NAME: DDPG
1.1: DDPG
TITLE  FetchReach
layer info  [50, 50, 50, 1]
layer info  [50, 50, 50, 1]
layer info  [50, 50, 50, 1]
layer info  [50, 50, 50, 1]
layer info  [50, 50, 4]
layer info  [50, 50, 4]
layer info  [50, 50, 4]
layer info  [50, 50, 4]
{'Actor': {'learning_rate': 0.001, 
            'linear_hidden_units': [50, 50], 
            'final_layer_activation': 'TANH', 
            'batch_norm': False, 
            'tau': 0.01, 
            'gradient_clipping_norm': 5, 
            'output_activation': None, 
            'hidden_activations': 'relu', 
            'dropout': 0.0, 
            'initialiser': 'default', 
            'columns_of_data_to_be_embedded': [], 
            'embedding_dimensions': [], 
            'y_range': ()}, 

'Critic': {'learning_rate': 0.01, 
            'linear_hidden_units': [50, 50, 50], 
            'final_layer_activation': None, 
            'batch_norm': False, 
            'buffer_size': 30000, 
            'tau': 0.01, 
            'gradient_clipping_norm': 5, 
            'output_activation': None, 
            'hidden_activations': 'relu', 
            'dropout': 0.0, 
            'initialiser': 'default', 
            'columns_of_data_to_be_embedded': [], 
            'embedding_dimensions': [], 
            'y_range': ()}, 
'batch_size': 256, 'discount_rate': 0.9, 'mu': 0.0, 'theta': 0.15, 'sigma': 0.25, 'update_every_n_steps': 10, 'learning_updates_per_learning_session': 10, 'HER_sample_proportion': 0.8, 'clip_rewards': False}
RANDOM SEED  1972201567
 Episode 26, Score: -50.00, Max score seen: -50.00, Rolling score: -50.00, Max rolling score seen: -inf
Process finished with exit code 0