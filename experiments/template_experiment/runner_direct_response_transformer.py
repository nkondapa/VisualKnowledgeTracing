import argparse
import time
import numpy as np
import pickle as pkl
from config import *
import sys
import utilities as ut
import experiment_utilities as eut
import config
import os

from tracing_models.transformer import DRTransformerSAINT

parser = argparse.ArgumentParser(description='Runner Recurrent Model')

# EXPERIMENT SPECIFICS
parser.add_argument('--experiment-data-name', type=str, default='default',
                    help='name of the source data for the learner')
parser.add_argument('--experiment-data-specifier', type=str, default='default',
                    help='name of the specific learner data file (usually some specific parameters)')
parser.add_argument('--experiment-group-name', type=str, default='default',
                    help='top level experiment group name (ie checkpoint_xyz)')
parser.add_argument('--experiment-name', type=str, default='default',
                    help='the experiment name (can run multiple experiment within an experiment group)')
parser.add_argument('--experiment-save-specifier', type=str, default='default',
                    help='the specific name to save the trained model under. The final path will be '
                         '"experiment-group-name"/"experiment-name"_"data-name"/"experiment-save-specifier"')
parser.add_argument('--feature-extractor-path', type=str, default='none',
                    help='the path to the pretrained feature extractor, default will instantiate a random one')
parser.add_argument('--feature-extractor-filename', type=str, default='none',
                    help='the above path leads to a folder where multiple models may be stored, '
                         'this will specify which one to pick')

# ARCHITECTURE PARAMS
parser.add_argument('--feature-dimension', type=int, default=16,
                    help='the dimensionality of the feature space')
parser.add_argument('--hidden-state-dimension', type=int, default=256,
                    help='the dimensionality of the lstm hidden state space')
parser.add_argument('--num-layers', type=int, default=3,
                    help='number of lstm layers')
parser.add_argument('--init-loss-weight', type=float, default=0.0,
                    help='whether to add extra weighting to fitting the initial position')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--information-level', type=str, default='argmax',
                    help='amount of information in learner responses')
parser.add_argument('--label-smoothing', type=float, default=0.0,
                    help='amount of label smoothing to apply')
parser.add_argument('--feature-extractor-architecture', type=str, default='none',
                    help='the feature extractor architecture to use if there is no specific pretrained model passed in')
parser.add_argument('--freeze-feature-extractor', action='store_true')
parser.add_argument('--num_encoder_layers', default=2)
parser.add_argument('--num_decoder_layers', default=2)

# TRAINING PARAMS
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--continue-training', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--early-stopping', action='store_true')
parser.add_argument('--cuda-online', action='store_true',
                    help='run the model by loading images to cuda when needed and not all at once')

# OPTIMIZER PARAMS
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--ip-lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--fe-lr', type=float, default=0.001,
                    help='initial learning rate')

# LOGGING/OUTPUT PARAMS
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='report interval')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        selected_device = 'cpu'
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        selected_device = 'cuda'
else:
    selected_device = 'cpu'

if args.cuda and not args.cuda_online:
    load_method = ('cuda', None)
elif args.cuda and args.cuda_online:
    load_method = ('cpu', 'cuda')
else:
    load_method = ('cpu', 'cpu')
###############################################################################
# LOAD DATA
args.data_folder = os.path.join(args.experiment_data_name, args.experiment_data_specifier)
learner_data, experiment_vars, training_data = eut.load_learner_data(args.data_folder, get_training_data=True,
                                                                     device='cpu')
mask_dict = experiment_vars['mask_dict']
split_dict = experiment_vars['split_dict']
evars = argparse.Namespace(**experiment_vars['input_params'])
input_data_dict = eut.format_input_data(learner_data, mask_dict, evars.sequence_length, **training_data)

###############################################################################
# Build the model
###############################################################################


if not args.continue_training:
    if args.feature_extractor_path != 'none':
        _, fe_model = ut.feature_extractors.load_feature_extractor(args.feature_extractor_path,
                                                                   args.feature_extractor_filename,
                                                                   device=config.device, eval=True)
        args.feature_dimension = fe_model.dimension
        if hasattr(fe_model, 'transform_data'):
            input_data_dict['input_data'] = fe_model.transform_data(input_data_dict['input_data'])
    else:
        if args.feature_extractor_architecture == 'none':
            if 'greebles' in args.experiment_data_specifier:
                fe_model = 'cnn128'
            elif 'butterflies' in args.experiment_data_name:
                fe_model = 'cnn144'
            elif 'oct' in args.experiment_data_name.lower():
                fe_model = 'cnn144_1ch'
        else:
            fe_model = args.feature_extractor_architecture

    model = DRTransformerSAINT(response_dimension=(evars.num_supervised_queries, evars.num_classes),
                               input_dimension=(evars.num_supervised_queries, args.feature_dimension),
                               out_dimension=(evars.num_classes, args.feature_dimension),
                               seq_len=evars.sequence_length,
                               pretrained_feature_extractor=fe_model,
                               num_encoder_layers=args.num_encoder_layers,
                               num_decoder_layers=args.num_decoder_layers,
                               hidden_state_dimension=args.hidden_state_dimension,
                               init_loss_weight=args.init_loss_weight,
                               label_smoothing=args.label_smoothing,
                               dropout=args.dropout,
                               information_level=args.information_level)
    model.to(selected_device)
    args.start_epoch = 1
else:
    folder = f'{args.experiment_group_name}/{args.experiment_name}/{args.experiment_save_specifier}/'
    model_dict = ut.file_utilities.load(folder, 'last.pt', return_dict=True)
    model = model_dict['model'].to(selected_device)
    if hasattr(model.feature_extractor, 'transform_data'):
        input_data_dict['input_data'] = model.feature_extractor.transform_data(input_data_dict['input_data'])
    args.start_epoch = model_dict['epoch']

formatted_data = model.format_data(**input_data_dict, mask_dict=mask_dict, split_dict=split_dict,
                                   learner_data=learner_data, device=load_method[0])

if args.freeze_feature_extractor:
    model.set_feature_extractor_trainable(False)

optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': args.lr},
    {'params': model.decoder.parameters(), 'lr': args.lr},
    {'params': model.out.parameters(), 'lr': args.lr},
    {'params': model.feature_extractor.parameters(), 'lr': args.fe_lr},
])
model_folder = f'{args.experiment_group_name}/{args.experiment_name}/{args.experiment_save_specifier}/'
print(model_folder)

###############################################################################
# Training code
###############################################################################

loss = None
results = {'acc': [],
           'train_per_epoch_loss': [],
           'train_per_epoch_init_loss': [],
           'train_loss': [],
           'train_per_sequence_per_learner_acc': [],
           'val_per_sequence_per_learner_acc': [],
           'test_per_sequence_per_learner_acc': [],
           'lr_sequence': [],
           'val_per_epoch_loss': [],
           'val_per_epoch_init_loss': [],
           'test_loss': None,
           'test_init_loss': None}


def get_batch(source, target, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = target[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(_data):
    # Turn on evaluation mode which disables dropout.

    _num_learners = _data['responses_t'].shape[0]
    model.eval()
    total_loss = 0.
    init_loss = 0.
    results_mat = torch.zeros(_num_learners, evars.sequence_length, evars.num_queries)
    batch_size = args.batch_size
    with torch.no_grad():
        num_batches = int(np.ceil(_num_learners / batch_size))
        for batch in range(num_batches):
            si = batch * batch_size
            ei = min((batch + 1) * batch_size, _num_learners)
            _bli = np.arange(si, ei)
            bsz = _bli.shape[0]

            model.zero_grad()
            data_batch = eut.index_dict_by_batch_indices(_data, _bli, load_method[1])
            out = model(**data_batch)

            results_mat[_bli, :] = (torch.argmax(out['responses'], dim=-1) == torch.argmax(data_batch['responses_t'],
                                                                                           dim=-1)).detach().cpu().type(
                torch.float).reshape(bsz, evars.sequence_length, evars.num_queries)

            loss, loss_vals = model.loss_function(out, data_batch['responses_t'])
            total_loss += loss.item()
            init_loss += loss_vals['init_loss']

    return total_loss / (_num_learners / batch_size), init_loss / (_num_learners / batch_size), results_mat


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_init_loss = 0.
    loss_denom = 0
    curr_loss_denom = 0
    start_time = time.time()
    data = formatted_data['train']
    _num_learners = data['responses_t'].shape[0]
    batch_size = args.batch_size
    num_batches = int(np.ceil(_num_learners / batch_size))

    learner_inds = np.arange(_num_learners)
    np.random.shuffle(learner_inds)
    batch_indices = list(range(num_batches))

    results_mat = torch.zeros(_num_learners, evars.sequence_length, evars.num_queries)
    epoch_loss = []
    epoch_init_loss = []
    for c, batch in enumerate(batch_indices):

        si = batch * batch_size
        ei = min((batch + 1) * batch_size, _num_learners)
        bli = learner_inds[si:ei]
        bsz = bli.shape[0]

        model.zero_grad()
        data_batch = eut.index_dict_by_batch_indices(data, bli, load_method[1])
        out = model(**data_batch)

        results_mat[bli, :] = (torch.argmax(out['responses'], dim=-1) == torch.argmax(data_batch['responses_t'],
                                                                                      dim=-1)).detach().cpu().type(
            torch.float).reshape(bsz, evars.sequence_length, evars.num_queries)

        loss, loss_vals = model.loss_function(out, data_batch['responses_t'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(bli)
        total_init_loss += loss_vals['init_loss'] * len(bli)
        curr_loss_denom += len(bli)
        loss_denom += len(bli)

        if c % args.log_interval == 0 and c > 0:
            cur_loss = total_loss / curr_loss_denom
            cur_init_loss = total_init_loss / curr_loss_denom
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | ms/batch {:5.2f} | '
                  'loss {:5.4f} | init_loss {:5.4f} | acc {:0.4f} '.format(
                epoch, batch, num_batches, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_init_loss,
                results_mat[bli, :].mean()))
            total_loss = 0
            total_init_loss = 0
            curr_loss_denom = 0
            start_time = time.time()

        if results['acc'] is None:
            results['acc'] = [results_mat[bli, :].mean().item()]
            results['train_loss'] = [loss.item()]
        else:
            results['acc'].append(results_mat[bli, :].mean().item())
            results['train_loss'].append(loss.item())

        epoch_loss.append(loss.item() * len(bli))
        epoch_init_loss.append(loss_vals['init_loss'] * len(bli) * len(bli))
        if args.dry_run:
            break

    results['train_per_sequence_per_learner_acc'].append(results_mat)
    results['train_per_epoch_loss'].append(np.sum(epoch_loss) / loss_denom)
    results['train_per_epoch_init_loss'].append(np.sum(epoch_init_loss) / loss_denom)


# Loop over epochs.
lr = args.lr
best_val_loss = None
best_val_acc = None
val_loss_increasing_counter = 0
checkpoint_list = [0, 200, 400]

# At any point you can hit Ctrl + C to break out of training early.
try:

    val_loss, val_init_loss, val_seq_acc = evaluate(formatted_data['val'])
    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        if epoch in checkpoint_list:
            ut.file_utilities.save_model({'epoch': epoch, 'model': model,
                                          'val_loss': val_loss, 'val_acc': torch.mean(val_seq_acc)},
                                         model_folder, f'checkpoint_{epoch}.pt')
        train()
        val_loss, val_init_loss, val_seq_acc = evaluate(formatted_data['val'])
        results['val_per_epoch_loss'].append(val_loss)
        results['val_per_epoch_init_loss'].append(val_init_loss)
        results['val_per_sequence_per_learner_acc'].append(val_seq_acc)
        results['lr_sequence'].append(lr)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.4f} | init loss {:5.4f} | '
              ' acc {:0.4f} '.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, val_init_loss, torch.mean(val_seq_acc).item()))

        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            ut.file_utilities.save_model(
                {'epoch': epoch, 'model': model, 'val_loss': val_loss, 'val_acc': torch.mean(val_seq_acc).item()},
                model_folder, 'best.pt')
            best_val_loss = val_loss

        if not best_val_acc or val_seq_acc.mean().item() > best_val_acc:
            ut.file_utilities.save_model(
                {'epoch': epoch, 'model': model, 'val_loss': val_loss, 'val_acc': torch.mean(val_seq_acc).item()},
                model_folder, 'best_acc.pt')
            best_val_acc = val_seq_acc.mean().item()

        if args.early_stopping:
            if val_loss > best_val_loss:
                val_loss_increasing_counter += 1
            else:
                val_loss_increasing_counter = 0
            if val_loss_increasing_counter >= 35:
                print('Stopping early due to val loss increasing')
                raise KeyboardInterrupt

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

ut.file_utilities.save_model({'epoch': epoch, 'model': model, 'val_loss': val_loss,
                              'val_acc': torch.mean(val_seq_acc).item()}, model_folder, 'last.pt')

# Run on test data.
test_loss, test_init_loss, test_seq_acc = evaluate(formatted_data['test'])
ut.file_utilities.save_model(
    {'epoch': epoch, 'model': model, 'test_loss': test_loss, 'test_acc': torch.mean(test_seq_acc).item()}, model_folder,
    'last.pt')

print('=' * 89)
print('| End of training | test loss {:5.2f}'.format(
    test_loss))
print(test_seq_acc.mean())
print(test_seq_acc.shape)
print(test_seq_acc.mean(0))
print('=' * 89)

results['test_loss'] = test_loss
results['test_init_loss'] = test_init_loss
results['test_per_sequence_per_learner_acc'].append(test_seq_acc)
ut.file_utilities.save_model(args, model_folder, 'args.pkl')
results_path = f'{config.ROOT}/experiments/{args.experiment_group_name}/experiment_results/{args.experiment_name}/{args.experiment_save_specifier}/'
os.makedirs(results_path, exist_ok=True)
results['args'] = args
results['last_epoch'] = epoch
path = results_path + '/results.pkl'
with open(path, 'wb') as f:
    pkl.dump(results, f)
