# Action Plan: League of Legends Pick/Ban Outcome Predictor

## Project Overview
Personal project to predict professional League of Legends pick and ban outcomes using machine learning and historical data.

## Current Status ✅ **PHASE 1 COMPLETED (Jan 2025)**
- ✅ **Oracle's Elixir Integration**: Full data pipeline with 77,412 training samples
- ✅ **Advanced Model**: DraftMLP with champion embeddings and multi-context encoding  
- ✅ **Rich Features**: Player stats, team preferences, meta analysis, fearless draft detection
- ✅ **Persistence Architecture**: FileRepository with parquet caching and incremental processing
- ✅ **Full Draft Sequences**: 10 picks + 10 bans + series context (vs. previous 6 bans + 1 pick)
- ✅ **Training Pipeline**: Successfully training with 50.6% validation accuracy (baseline ~0.6%)
- **Project constraints:** Solo developer, free time only, minimal hosting budget

## Action Plan for Full System Implementation

### ✅ Phase 1: Data and Model Development **COMPLETED**
1. **✅ Data Collection & Processing** 
   - ✅ Oracle's Elixir CSV integration with 77,412 samples
   - ✅ Player performance data (KDA, champion pick rates, recent form)  
   - ✅ Team performance metrics and patch/meta data
   - ✅ **Fearless draft detection and series grouping implemented**
   - ✅ Full draft sequences (10 picks + 10 bans + series context)
   - ✅ Effective ban pools (up to 50 champions in fearless Bo5)
   - ✅ Feature engineering pipeline with pandas (25+ functions)
   - ✅ Data validation and quality checks

2. **✅ Model Development**
   - ✅ DraftMLP with champion embeddings and multi-context architecture
   - ✅ Training pipeline with Oracle's Elixir rich features
   - ✅ Evaluation metrics and validation (50.6% accuracy vs ~0.6% baseline)
   - ✅ Variable-length effective ban pools handled via feature engineering
   - **Next**: Hyperparameter tuning and advanced architectures

### Phase 2: Simple Web Interface **NEXT**
3. **Basic Web UI** (Deploy to free hosting)
   - Simple interface for inputting draft scenarios
   - **Support for fearless draft mode selection and series context**
   - Display prediction results and confidence  
   - Basic visualization of model insights
   - Show effective ban pool and available champions
   - **Prerequisites**: ✅ Model training, ✅ Data pipeline, ✅ Inference capabilities

### Phase 3: Data Infrastructure (When file management becomes painful)
4. **Database Integration**
   - Move to database when >100k records or complex joins needed
   - Design schema for players, teams, matches, patches, **series**
   - **Optimize for fearless draft queries (series-based lookups)**

5. **Automated Data Collection**
   - Implement scripts for periodic data fetching
   - Basic error handling and validation

### Phase 4: Enhancements (If project gains traction)
6. **Advanced Features**
   - Model comparison capabilities
   - Historical performance tracking
   - Enhanced visualizations

## Success Metrics
- ✅ **Model accuracy improvement**: 50.6% validation accuracy (vs ~0.6% random baseline)
- ✅ **Personal learning**: Advanced ML pipeline, data engineering, fearless draft analysis
- ✅ **Working prototype**: Oracle's Elixir integration with 77,412 samples
- **Next**: Web interface for portfolio demonstration

## Timeline Considerations
- Focus on momentum and quick wins
- Move to next phase only when current approach becomes limiting
- Prioritize features that provide immediate value
