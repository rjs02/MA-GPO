# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/trainer/rm_trainer.py

from abc import ABC
import os 
import shutil
import loralib as lora
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from general_preference.utils.custom_distributed_sampler import GroupDistributedSampler
from tqdm import tqdm
import deepspeed
from general_preference.models import PairWiseLoss, GeneralPreferenceLoss, HighDimGeneralPreferenceLoss, SFTMeanLoss, SFTSumLoss, DPORefFreeLoss, SFTVanillaLoss
from general_preference.models import GeneralPreferenceLearnableTauLoss, GeneralPreferenceLearnableTauRegressionLoss, GeneralPreferenceRegressionLoss
from general_preference.models import PairWiseLearnableTauLoss, PairWiseLearnableTauRegressionLoss, PairWiseRegressionLoss, HighDimGeneralPreferenceRegressionMoELoss
from general_preference.models import HighDimGeneralPreferenceRegressionLoss, HighDimGeneralPreferenceMoELoss
class GeneralPreferenceRewardTrainer(ABC):
    """
        Trainer to use while training reward model.

    Args:
        model (torch.nn.Module): the model to train
        strategy (Strategy): the strategy to use for training
        optim(Optimizer): the optimizer to use for training
        train_dataset (RewardDataset): the dataset to use for training
        eval_dataset (RewardDataset): the dataset to use for evaluation
        batch_size (int, defaults to 1): the batch size while training
        max_epochs (int, defaults to 2): the number of epochs to train
        is_general_preference (bool, defaults to False): Whether the model is a General Preference model.
        tau (float, defaults to 0.1): Hyperparameter tau used in the calculation of General Preference loss.
        value_head_dim (int, defaults to 2): Dimension of the value head in the General Preference model. Ignored by the Bradley Terry model.

    """

    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        tokenizer,
        max_epochs: int = 2,
        is_general_preference: bool = False,
        tau: float = 0.1,
        value_head_dim: int = 2,
        
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.optimizer = optim
        self.tokenizer = tokenizer
        self.args = strategy.args
        self.is_general_preference = is_general_preference

        if is_general_preference:
            if value_head_dim == 2 and not self.args.add_prompt_head:
                self.loss_fn = GeneralPreferenceLoss(tau)
                self.strategy.print("GeneralPreference Loss")
                # self.loss_fn = GeneralPreferenceRegressionLoss(tau, self.args.regression_target_margin)
                # self.loss_fn = GeneralPreferenceLearnableTauLoss()
                # self.loss_fn = GeneralPreferenceLearnableTauRegressionLoss(target_margin=self.args.regression_target_margin)
            else:
                assert value_head_dim % 2 == 0, "Dimension of value head for general preference model can not be odd!"
                if self.args.add_prompt_head:
                    self.loss_fn = HighDimGeneralPreferenceMoELoss(model=self.model, value_head_dim=value_head_dim, softmax_tau=tau)
                    # self.loss_fn = HighDimGeneralPreferenceRegressionMoELoss(model=self.model, value_head_dim=value_head_dim, target_margin=self.args.regression_target_margin, softmax_tau=tau)
                else:
                    self.loss_fn = HighDimGeneralPreferenceLoss(tau, value_head_dim)
                    # strategy.print("Loss for high-dimensional value head General Preference model.")
                    # self.loss_fn = HighDimGeneralPreferenceRegressionLoss(tau=tau, target_margin=self.args.regression_target_margin, value_head_dim=value_head_dim)    
        else:
            self.loss_fn = PairWiseLoss(tau)
            self.strategy.print("LogSigmoid Loss")
            # self.loss_fn = PairWiseRegressionLoss(tau, self.args.regression_target_margin)
            # self.loss_fn = PairWiseLearnableTauLoss()
            # self.loss_fn = PairWiseLearnableTauRegressionLoss(target_margin=self.args.regression_target_margin)

        # self.ptx_loss_fn = SFTVanillaLoss()
        # self.ptx_loss_fn = SFTMeanLoss(self.args.reward_scaler_beta)
        self.ptx_loss_fn = SFTSumLoss(self.args.reward_scaler_beta)
        # self.ptx_loss_fn = DPORefFreeLoss(self.args.reward_scaler_beta, self.args.reward_margin)
        

        self.margin_loss = self.strategy.args.margin_loss
        self.compute_fp32_loss = self.strategy.args.compute_fp32_loss

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args, consumed_samples=0):
        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()  # Evaluate once per epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        eval_loss_minimum = None
        # Resume from checkpoint if consumed_samples > 0
        global_step = consumed_samples + 1
        start_epoch = consumed_samples // self.train_dataloader.__len__()
        steps_in_current_epoch = consumed_samples % self.train_dataloader.__len__()
        
        if consumed_samples > 0:
            self.strategy.print(f"Resuming from epoch {start_epoch}, step {steps_in_current_epoch} (global_step={global_step})")
        
        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        for epoch in range(start_epoch, self.epochs):
            #  train
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )
                
            if isinstance(self.train_dataloader.sampler, GroupDistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            self.model.train()
            loss_mean = 0
           
            for step_in_epoch, (chosen_ids, c_mask, reject_ids, r_mask, margin, chosen_response_len, rejected_response_len) in enumerate(self.train_dataloader):
                # Skip already-trained steps when resuming
                if epoch == start_epoch and step_in_epoch < steps_in_current_epoch:
                    continue
                    
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                chosen_response_len_tensor = torch.tensor(chosen_response_len).view(-1, 1).to(torch.cuda.current_device())
                rejected_response_len_tensor = torch.tensor(rejected_response_len).view(-1, 1).to(torch.cuda.current_device())
                
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                
                return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss) else False
                chosen_reward, reject_reward, outputs = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, return_output
                )
                # loss function
                if self.compute_fp32_loss:
                    chosen_reward = chosen_reward.float()
                    reject_reward = reject_reward.float()

                if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) or isinstance(self.loss_fn, HighDimGeneralPreferenceMoELoss):
                    chosen_last_hidden_states = outputs["last_hidden_state"][: chosen_ids.shape[0], :, :]
                    prompt_end_index = chosen_last_hidden_states.size(1) - chosen_response_len_tensor - 1
                    prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_end_index_expanded).squeeze(1)
                    preference_loss, prob, accuracy = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state.to(torch.cuda.current_device()), margin)
                else:
                    preference_loss, prob, accuracy = self.loss_fn(chosen_reward, reject_reward, margin)
                # Binary NLL for positive pair labels (chosen should beat rejected).
                nll = -torch.log(prob.clamp(min=1e-8, max=1 - 1e-8))
                
                if args.add_pretrain_loss:
                    if isinstance(self.ptx_loss_fn, DPORefFreeLoss):
                        chosen_output = self.model.forward(chosen_ids, attention_mask=c_mask)
                        chosen_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        chosen_log_probs = chosen_output["logits"]
                        rejected_output = self.model.forward(reject_ids, attention_mask=r_mask)
                        rejected_label = torch.where(
                            r_mask.bool(),
                            reject_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        rejected_log_probs = rejected_output["logits"] 
                        chosen_reward_ptx_loss = self.ptx_loss_fn(chosen_log_probs, chosen_label, c_mask.bool(), rejected_log_probs, rejected_label, r_mask.bool())
                    else:
                        ptx_output = self.model.forward(chosen_ids, attention_mask=c_mask)
                        ptx_label = torch.where(
                            c_mask.bool(),
                            chosen_ids,
                            self.ptx_loss_fn.IGNORE_INDEX,
                        ).to(torch.cuda.current_device())
                        ptx_log_probs = ptx_output["logits"]
                        chosen_reward_ptx_loss = self.ptx_loss_fn(ptx_log_probs, ptx_label, c_mask.bool())
                    
                    loss = (1 - args.ptx_loss_coef) * preference_loss + chosen_reward_ptx_loss * args.ptx_loss_coef
                else:
                    loss = preference_loss

                
                self.strategy.backward(loss, self.model, self.optimizer)
                
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                loss_mean = loss_mean * 0.9 + 0.1 * preference_loss.item()
                
                # Compute response length statistics for the batch
                chosen_len_batch = chosen_response_len_tensor.float()
                rejected_len_batch = rejected_response_len_tensor.float()

                # Get current learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                logs_dict = {
                    "preference_loss": preference_loss.item(),
                    "nll": nll.item(),
                    "prob": prob.item(),
                    "accuracy": accuracy.item(),
                    "loss_mean": loss_mean,
                    "lr": current_lr,
                    "chosen_response_len_avg": chosen_len_batch.mean().item(),
                    "rejected_response_len_avg": rejected_len_batch.mean().item(),
                    "chosen_response_len_std": chosen_len_batch.std().item() if chosen_len_batch.numel() > 1 else 0.0,
                    "rejected_response_len_std": rejected_len_batch.std().item() if rejected_len_batch.numel() > 1 else 0.0,
                }
                    
                # logs/checkpoints/evaluate
                eval_loss_minimum = self.save_logs_and_checkpoints(args, global_step, step_bar, eval_loss_minimum, logs_dict)
                torch.distributed.barrier()
                step_bar.update()
                global_step += 1
                
            epoch_bar.update()

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()

    # logs/checkpoints/evaluate
    def save_logs_and_checkpoints(self, args, global_step, step_bar, eval_loss_minimum, logs_dict={}):
        if global_step % args.logging_steps == 0:
            # step bar
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)
            # wandb
            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)
        # eval
        if global_step % args.eval_steps == 0 and self.eval_dataloader:
            try:
                eval_loss = self.evaluate(self.eval_dataloader, global_step)      
                torch.distributed.barrier()
            except Exception as e:
                self.strategy.print(f"Error during evaluation: {str(e)}")
                raise
            if args.save_best_model:
                save_path = os.path.join(args.save_path, f"step_{global_step}")
                if eval_loss_minimum is None or eval_loss < eval_loss_minimum:
                    try:
                        self.strategy.save_model(self.model, self.tokenizer, save_path)
                        eval_loss_minimum = eval_loss
                    except Exception as e:
                        self.strategy.print(f"Error during saving model: {str(e)}")
                        raise
                    try:
                        if self.strategy.is_rank_0():  
                            self.clean_old_checkpoints(args.save_path, args.save_best_model) 
                    except Exception as e:
                        self.strategy.print(f"Error during deleting old checkpoint: {str(e)}")
                        raise
                                     
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step_{global_step}"
            
            # Save full DeepSpeed checkpoint (for resuming training with optimizer/scheduler states)
            # This creates: ckpt_path/tag/mp_rank_00_model_states.pt, etc.
            self.strategy.save_ckpt(
                self.model, 
                args.ckpt_path, 
                tag, 
                args.max_ckpt_num, 
                args.max_ckpt_mem,
                client_state={'consumed_samples': global_step - 1}
            )
            
            # Update 'latest' file to point to the newest checkpoint
            if self.strategy.is_rank_0():
                latest_file = os.path.join(args.ckpt_path, "latest")
                try:
                    with open(latest_file, 'w') as f:
                        f.write(tag)
                    self.strategy.print(f"✓ Updated latest checkpoint pointer: {tag}")
                except Exception as e:
                    self.strategy.print(f"⚠️  Warning: Failed to update 'latest' file: {e}")
            
            # Also save model weights separately (for easy loading/evaluation)
            # Only save if save_path is different from ckpt_path
            if args.save_path != args.ckpt_path:
                model_save_path = os.path.join(args.save_path, tag)
                self.strategy.save_model(self.model, self.tokenizer, model_save_path)
            else:
                self.strategy.print(f"⚠️  Skipping duplicate model save (save_path == ckpt_path)")
            
        if self.strategy.is_rank_0():  
            return eval_loss_minimum
    
    def clean_old_checkpoints(self, output_dir, max_checkpoints=3): 
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('step_')] # If the number of directories exceeds max_checkpoints, delete the oldest one 
        if len(subdirs) > max_checkpoints: 
            subdirs.sort(key=lambda x: int(x.split('_')[1])) 
            dir_to_delete = os.path.join(output_dir, subdirs[0]) 
            try:
                shutil.rmtree(dir_to_delete) 
            except Exception as e:
                print(f"Error deleting old checkpoint{dir_to_delete}: {e}")


    def evaluate(self, eval_dataloader, steps=0):
        step_bar = tqdm(
            range(eval_dataloader.__len__()),
            desc="Eval stage of steps %d" % steps,
            disable=not self.strategy.is_rank_0(),
        )
        
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            nll_sum = 0
            prob_sum = 0
            accuracy_sum = 0
            for chosen_ids, c_mask, reject_ids, r_mask, margin, chosen_response_len, rejected_response_len in eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
                c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
                reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
                r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
                if self.margin_loss:
                    margin = torch.tensor(margin).to(torch.cuda.current_device())
                else:
                    margin = None
                chosen_response_len_tensor = torch.tensor(chosen_response_len).view(-1, 1).to(torch.cuda.current_device())
                rejected_response_len_tensor = torch.tensor(rejected_response_len).view(-1, 1).to(torch.cuda.current_device())

                return_output = True if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss) else False
                chosen_reward, reject_reward, outputs = self.concatenated_forward(
                    self.model, chosen_ids, c_mask, reject_ids, r_mask, return_output
                )
                
                if isinstance(self.loss_fn, HighDimGeneralPreferenceRegressionMoELoss):
                    chosen_last_hidden_states = outputs["last_hidden_state"][: chosen_ids.shape[0], :, :]
                    prompt_len = chosen_last_hidden_states.size(1) - chosen_response_len_tensor
                    prompt_len_expanded = prompt_len.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
                    prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_len_expanded).squeeze(1)
                    preference_loss, prob, accuracy = self.loss_fn(chosen_reward, reject_reward, prompt_hidden_state, margin)
                else:
                    preference_loss, prob, accuracy = self.loss_fn(chosen_reward, reject_reward, margin)
                nll = -torch.log(prob.clamp(min=1e-8, max=1 - 1e-8))
                    
                loss = preference_loss

                loss_sum += loss.item()
                nll_sum += nll.item()
                prob_sum += prob.item()
                accuracy_sum += accuracy.item()
                  
                step_bar.update()

            loss_mean = loss_sum / self.eval_dataloader.__len__()
            nll_mean = nll_sum / self.eval_dataloader.__len__()
            prob_mean = prob_sum / self.eval_dataloader.__len__()
            accuracy_mean = accuracy_sum / self.eval_dataloader.__len__()

            bar_dict = {
                "eval_loss_mean": loss_mean,
                "nll_mean": nll_mean,
                "prob_mean": prob_mean,
                "accuracy_mean": accuracy_mean,
            }
            logs = self.strategy.all_reduce(bar_dict)
            step_bar.set_postfix(logs)

            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                self._wandb.log(logs)

        self.model.train()  # reset model state
        torch.cuda.empty_cache() 
        if self.strategy.is_rank_0():  
            return loss_mean

    def concatenated_forward(self, model, chosen_ids, c_mask, reject_ids, r_mask, return_output: bool = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        """
        input_ids, att_masks = self.concatenated_inputs(chosen_ids, c_mask, reject_ids, r_mask)
        all_values, outputs = model.custom_forward(input_ids, attention_mask=att_masks, return_output=return_output)
        chosen_rewards = all_values[: chosen_ids.shape[0]]
        rejected_rewards = all_values[chosen_ids.shape[0] :]
        
        return chosen_rewards, rejected_rewards, outputs
    
    

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """

        def pad_to_length(tensor, length, pad_value, dim=-1):
            if tensor.size(dim) >= length:
                return tensor
            else:
                pad_size = list(tensor.shape)
                pad_size[dim] = length - tensor.size(dim)
                # left pad
                return torch.cat(
                    [pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device), tensor], dim=dim
                )

        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, self.tokenizer.pad_token_id),
                pad_to_length(reject_ids, max_length, self.tokenizer.pad_token_id),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks
