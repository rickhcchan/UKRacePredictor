# AI Integration with WTG.AI.Prompts - Cargowise Domain

This document describes the integration of WTG.AI.Prompts cargowise domain functionality into the UK Race Predictor system.

## Overview

The UK Race Predictor now includes AI-powered analysis using prompts adapted from WiseTech Global's cargowise domain methodology. This integration applies supply chain logistics and optimization principles to horse racing prediction and analysis.

## Features

### 🤖 AI-Enhanced Race Analysis
- **Systematic Analysis**: Applies cargowise logistics methodology to race evaluation
- **Performance Supply Chain**: Analyzes horse form trends using supply chain reliability principles
- **Competitive Positioning**: Uses optimization algorithms for field strength assessment
- **Risk-Return Evaluation**: Applies cargo routing efficiency metrics to betting decisions

### 📊 Three Cargowise Domain Prompts

#### 1. Analysis Prompt (`analysis.prompt`)
Applies supply chain systematic evaluation to race data:
- Performance logistics assessment
- Form supply chain analysis  
- Course & distance optimization
- Competitive positioning within field
- Risk assessment using cargo routing principles

#### 2. Prediction Enhancement Prompt (`prediction.prompt`)
Enhances ML model predictions using forecasting methodologies:
- Demand forecasting principles for outcome likelihood
- Supply chain risk assessment for prediction variables
- Optimization algorithms for probability refinement
- Quality assurance checks against historical patterns

#### 3. Strategy Prompt (`strategy.prompt`)
Generates betting strategies using resource allocation principles:
- Strategic resource allocation across opportunities
- Portfolio diversification and risk management
- Supply chain reliability assessment of selections
- Performance optimization with downside risk control

## Usage

### Enable AI Analysis (Default)
```bash
# AI analysis is enabled by default
python scripts/predict_races.py --dry-run

# Explicitly enable AI analysis
python scripts/predict_races.py --ai --dry-run
```

### Disable AI Analysis
```bash
# Disable AI analysis
python scripts/predict_races.py --no-ai --dry-run
```

### Combined with Other Features
```bash
# AI analysis with odds fetching
python scripts/predict_races.py --ai --odds --dry-run

# AI analysis with specific models
python scripts/predict_races.py --model win,top3 --ai --dry-run

# AI analysis with custom strategy
python scripts/predict_races.py --strategy default --ai --dry-run
```

## Output Examples

### AI Analysis Display
When AI is enabled, each race includes cargowise methodology analysis:

```
📍 Ascot - 15:30 (12 horses total)

🤖 AI Analysis (Cargowise Methodology Applied)

**Race Logistics Assessment:**
- Course: Ascot
- Field Optimization: 12 horses (Manageable field size)
- Race Type: Handicap

**Performance Supply Chain Analysis:**
Based on cargowise systematic evaluation principles:

1. **Efficiency Metrics**: Top contenders show consistent delivery patterns
2. **Reliability Factors**: Recent form trends indicate stable performance corridors
3. **Environmental Adaptability**: Course and distance suitability analysis completed
4. **Competitive Positioning**: Relative strength assessment within current field composition
5. **Risk-Return Evaluation**: Probability distributions aligned with market efficiency

**Strategic Insights:**
- Apply systematic risk management protocols
- Consider field dynamics in position sizing
- Monitor real-time market adjustments
- Maintain portfolio balance principles

**Confidence Level**: Moderate to High (based on data quality and pattern recognition)
```

## Technical Implementation

### Directory Structure
```
prompts/
├── __init__.py
├── prompt_manager.py           # Prompt loading and formatting
├── ai_integration.py          # AI integration logic
└── domains/
    └── cargowise/
        ├── analysis.prompt    # Race analysis template
        ├── prediction.prompt  # Prediction enhancement template
        └── strategy.prompt    # Strategy recommendation template
```

### Key Components

#### PromptManager Class
- Loads prompt templates from cargowise domain
- Handles template formatting with race data
- Provides structured prompt generation methods

#### AIIntegration Class  
- Manages AI analysis workflow
- Integrates with existing prediction pipeline
- Provides fallback behavior when AI is disabled

#### Integration Points
- **Race Display**: AI analysis appears after race headers
- **Prediction Enhancement**: AI insights augment ML model outputs
- **Strategy Recommendations**: Cargowise planning principles guide betting decisions

## Cargowise Methodology Adaptation

### Supply Chain → Horse Racing Mapping

| Cargowise Concept | Racing Application |
|-------------------|-------------------|
| **Cargo Routing** | Course/distance suitability analysis |
| **Supply Chain Reliability** | Horse form consistency patterns |
| **Demand Forecasting** | Win probability prediction |
| **Resource Allocation** | Betting capital distribution |
| **Risk Management** | Portfolio exposure control |
| **Performance Optimization** | Expected value maximization |
| **Quality Assurance** | Prediction validation checks |

### Analysis Framework
1. **Efficiency Metrics**: Win rates, place rates, strike rates
2. **Reliability Factors**: Consistency, recent form trends
3. **Environmental Adaptability**: Course/going preferences
4. **Competitive Advantage**: Relative positioning in field
5. **Risk-Return Evaluation**: Probability vs. market odds

## Testing

### AI Integration Test Suite
```bash
# Run comprehensive AI integration tests
python scripts/test_ai_integration.py
```

Test coverage includes:
- Prompt manager functionality
- AI integration workflow
- Template formatting
- Error handling
- Status reporting

### Expected Test Output
```
🏇 UK Race Predictor - AI Integration Test Suite
============================================================
✅ Prompt Manager: PASSED
✅ AI Integration: PASSED  
✅ Prompt Formatting: PASSED
============================================================
Test Results: 3/3 tests passed
🎉 All tests passed! AI integration is working correctly.
```

## Configuration

### Enable/Disable AI
The AI integration can be controlled via command-line flags:
- `--ai`: Enable AI analysis (default: enabled)
- `--no-ai`: Disable AI analysis

### Error Handling
- **Graceful Degradation**: If AI components are unavailable, the system continues normal operation
- **Non-blocking**: AI analysis errors don't prevent race predictions
- **User Feedback**: Clear status messages indicate AI availability

## Benefits

### Enhanced Decision Making
- **Multiple Perspectives**: Combines ML predictions with AI analysis
- **Systematic Framework**: Applies proven logistics methodology
- **Risk Awareness**: Cargowise risk management principles
- **Strategic Depth**: Beyond simple probability calculations

### Professional Methodology
- **Established Framework**: Leverages WiseTech Global's proven approach
- **Industry Standards**: Applies supply chain best practices
- **Systematic Process**: Consistent analysis methodology
- **Quality Assurance**: Built-in validation and checking

## Future Enhancements

### Potential Extensions
- **Real-time AI Analysis**: Live race condition adjustments
- **Historical Pattern Recognition**: Deep learning on racing patterns
- **Multi-domain Integration**: Other WTG.AI.Prompts domains
- **Custom Prompt Development**: User-defined analysis templates

### Integration Opportunities
- **External AI Services**: OpenAI, Claude, or other LLM providers
- **Enhanced Data Sources**: More comprehensive racing data
- **Advanced Analytics**: Complex pattern recognition
- **Interactive Analysis**: User-guided AI exploration

## Troubleshooting

### Common Issues

#### "AI integration not available"
- Check prompts directory exists
- Verify cargowise domain prompts are present
- Ensure all required prompt files exist

#### "Cargowise prompt analysis disabled"
- Use `--ai` flag to enable
- Check prompts/domains/cargowise/ directory
- Verify prompt file format and content

#### AI analysis not appearing
- Ensure `--ai` flag is used (default: enabled)
- Check for error messages in output
- Verify race data is properly formatted

### Debug Information
```bash
# Test AI integration status
python scripts/test_ai_integration.py

# Run predictions with AI disabled for comparison
python scripts/predict_races.py --no-ai --dry-run

# Check prompt availability
python -c "from prompts.prompt_manager import prompt_manager; print(prompt_manager.list_available_prompts())"
```