import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class HumorFusionModelV2(nn.Module):
    def __init__(self, 
                 audio_input_dim, 
                 text_input_dim, 
                 video_au_input_dim,
                 audio_lstm_hidden_dim=128, 
                 video_au_lstm_hidden_dim=64,
                 text_fc_dim=256,
                 fusion_dim=256, 
                 num_classes=2, 
                 dropout_rate=0.3):
        super(HumorFusionModelV2, self).__init__()

        self.audio_input_dim = audio_input_dim
        self.text_input_dim = text_input_dim
        self.video_au_input_dim = video_au_input_dim

        # Audio Branch (LSTM)
        self.audio_lstm = nn.LSTM(audio_input_dim, audio_lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.audio_fc = nn.Linear(audio_lstm_hidden_dim * 2, fusion_dim // 3) # //3 for later concatenation

        # Text Branch (FC layers for pooled embeddings)
        self.text_fc1 = nn.Linear(text_input_dim, text_fc_dim)
        self.text_fc2 = nn.Linear(text_fc_dim, fusion_dim // 3)

        # Video AU Branch (LSTM)
        self.video_au_lstm = nn.LSTM(video_au_input_dim, video_au_lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.video_au_fc = nn.Linear(video_au_lstm_hidden_dim * 2, fusion_dim - 2*(fusion_dim//3) ) # Remaining part

        # Fusion and Classification
        self.fusion_fc1 = nn.Linear(fusion_dim, fusion_dim // 2)
        self.fusion_fc2 = nn.Linear(fusion_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        logger.info("HumorFusionModelV2 initialized.")
        logger.info(f"  Audio Input Dim: {audio_input_dim}, LSTM Hidden: {audio_lstm_hidden_dim} (bi)")
        logger.info(f"  Text Input Dim: {text_input_dim}, FC Dim: {text_fc_dim}")
        logger.info(f"  Video AU Input Dim: {video_au_input_dim}, LSTM Hidden: {video_au_lstm_hidden_dim} (bi)")
        logger.info(f"  Fusion Dim: {fusion_dim}, Num Classes: {num_classes}, Dropout: {dropout_rate}")


    def forward(self, audio_features, text_features, video_au_features):
        # audio_features: (batch_size, seq_len_audio, audio_input_dim)
        # text_features: (batch_size, text_input_dim)
        # video_au_features: (batch_size, seq_len_video, video_au_input_dim)

        # Audio Branch
        # Check if audio_features are all padding (e.g., if max_audio_len was used and sample was short)
        # A more robust check might involve sequence lengths if provided by dataloader
        if audio_features.abs().sum() == 0: # A simple check for all zeros
            x_audio = torch.zeros(audio_features.size(0), self.audio_fc.out_features, device=audio_features.device)
        else:
            _, (h_audio, _) = self.audio_lstm(audio_features)
            # Concatenate hidden states from both directions
            x_audio = torch.cat((h_audio[-2,:,:], h_audio[-1,:,:]), dim=1)
            x_audio = self.dropout(x_audio)
            x_audio = F.relu(self.audio_fc(x_audio))

        # Text Branch
        x_text = F.relu(self.text_fc1(text_features))
        x_text = self.dropout(x_text)
        x_text = F.relu(self.text_fc2(x_text))

        # Video AU Branch
        # Check if video_au_features are all padding (e.g., if OpenFace CSV was missing or empty)
        if video_au_features.abs().sum() == 0: # A simple check for all zeros
            x_video_au = torch.zeros(video_au_features.size(0), self.video_au_fc.out_features, device=video_au_features.device)
        else:
            # Check if sequence length is 0 (can happen if placeholder was (0, num_aus))
            if video_au_features.shape[1] == 0:
                 x_video_au = torch.zeros(video_au_features.size(0), self.video_au_fc.out_features, device=video_au_features.device)
            else:
                _, (h_video_au, _) = self.video_au_lstm(video_au_features)
                x_video_au = torch.cat((h_video_au[-2,:,:], h_video_au[-1,:,:]), dim=1)
                x_video_au = self.dropout(x_video_au)
                x_video_au = F.relu(self.video_au_fc(x_video_au))
        
        # Fusion
        # Concatenate the outputs of the three branches
        x_fused = torch.cat((x_audio, x_text, x_video_au), dim=1)
        x_fused = self.dropout(x_fused)
        
        x_fused = F.relu(self.fusion_fc1(x_fused))
        x_fused = self.dropout(x_fused)
        
        output = self.fusion_fc2(x_fused) # Raw logits, CrossEntropyLoss will apply softmax
        
        return output

if __name__ == '__main__':
    # Example Usage (for testing the model class)
    batch_size = 4
    audio_seq_len = 100
    video_seq_len = 80
    
    # Dimensions from feature extraction
    # These should match the output dimensions of your feature extractors
    # For WavLM-Base+: 1024 (if using last hidden state mean pooled per segment)
    # or (num_segments, 1024) if sequential. Let's assume sequential input here.
    AUDIO_DIM = 1024 
    TEXT_DIM = 1024  # XLM-R Large
    VIDEO_AU_DIM = 17 # Number of AU intensity columns

    # Model Hyperparameters
    AUDIO_LSTM_HIDDEN = 256
    VIDEO_AU_LSTM_HIDDEN = 128
    TEXT_FC_DIM_MODEL = 512
    FUSION_OUTPUT_DIM = 512 # This is the combined dim before final classification layers
    NUM_CLASSES = 2
    DROPOUT = 0.5

    model = HumorFusionModelV2(
        audio_input_dim=AUDIO_DIM,
        text_input_dim=TEXT_DIM,
        video_au_input_dim=VIDEO_AU_DIM,
        audio_lstm_hidden_dim=AUDIO_LSTM_HIDDEN,
        video_au_lstm_hidden_dim=VIDEO_AU_LSTM_HIDDEN,
        text_fc_dim=TEXT_FC_DIM_MODEL,
        fusion_dim=FUSION_OUTPUT_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT
    )

    # Create dummy input tensors
    dummy_audio = torch.randn(batch_size, audio_seq_len, AUDIO_DIM)
    dummy_text = torch.randn(batch_size, TEXT_DIM)
    dummy_video_au = torch.randn(batch_size, video_seq_len, VIDEO_AU_DIM)
    
    # Test forward pass
    logger.info("Testing model forward pass with random data...")
    try:
        predictions = model(dummy_audio, dummy_text, dummy_video_au)
        logger.info(f"Output shape: {predictions.shape}") # Expected: (batch_size, num_classes)
        assert predictions.shape == (batch_size, NUM_CLASSES)
        logger.info("Forward pass successful.")
    except Exception as e:
        logger.error(f"Error during model forward pass: {e}", exc_info=True)

    # Test with zero video input (simulating missing OpenFace data)
    logger.info("Testing model forward pass with zero video AU data...")
    dummy_video_au_zeros = torch.zeros(batch_size, video_seq_len, VIDEO_AU_DIM)
    try:
        predictions_zero_video = model(dummy_audio, dummy_text, dummy_video_au_zeros)
        logger.info(f"Output shape (zero video): {predictions_zero_video.shape}")
        assert predictions_zero_video.shape == (batch_size, NUM_CLASSES)
        logger.info("Forward pass with zero video AU data successful.")
    except Exception as e:
        logger.error(f"Error during model forward pass with zero video AU: {e}", exc_info=True)

    # Test with zero audio input
    logger.info("Testing model forward pass with zero audio data...")
    dummy_audio_zeros = torch.zeros(batch_size, audio_seq_len, AUDIO_DIM)
    try:
        predictions_zero_audio = model(dummy_audio_zeros, dummy_text, dummy_video_au)
        logger.info(f"Output shape (zero audio): {predictions_zero_audio.shape}")
        assert predictions_zero_audio.shape == (batch_size, NUM_CLASSES)
        logger.info("Forward pass with zero audio data successful.")
    except Exception as e:
        logger.error(f"Error during model forward pass with zero audio: {e}", exc_info=True)
