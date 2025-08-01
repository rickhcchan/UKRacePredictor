"""
Prompt Manager for UK Race Predictor
Handles loading and processing of AI prompts from the WTG.AI.Prompts format
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class PromptManager:
    """Manages AI prompts adapted from @WiseTechGlobal/WTG.AI.Prompts format"""
    
    def __init__(self, base_path: Optional[str] = None):
        if base_path is None:
            # Default to prompts directory relative to this file
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        
    def load_prompt(self, domain: str, prompt_name: str) -> Optional[str]:
        """
        Load a prompt template from the specified domain
        
        Args:
            domain: Domain name (e.g., 'cargowise')
            prompt_name: Prompt file name without extension (e.g., 'analysis')
            
        Returns:
            Prompt template string or None if not found
        """
        prompt_path = self.base_path / "domains" / domain / f"{prompt_name}.prompt"
        
        try:
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                self.logger.info(f"Loaded prompt: {domain}/{prompt_name}")
                return content
            else:
                self.logger.warning(f"Prompt not found: {prompt_path}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading prompt {domain}/{prompt_name}: {e}")
            return None
    
    def format_prompt(self, template: str, **kwargs) -> str:
        """
        Format a prompt template with provided variables
        
        Args:
            template: Prompt template string
            **kwargs: Variables to substitute in template
            
        Returns:
            Formatted prompt string
        """
        try:
            return template.format(**kwargs)
        except KeyError as e:
            self.logger.error(f"Missing variable in prompt template: {e}")
            return template
        except Exception as e:
            self.logger.error(f"Error formatting prompt: {e}")
            return template
    
    def get_cargowise_analysis_prompt(self, race_data: Dict[str, Any]) -> Optional[str]:
        """
        Get formatted analysis prompt using cargowise domain template
        
        Args:
            race_data: Dictionary containing race information
            
        Returns:
            Formatted prompt string or None
        """
        template = self.load_prompt("cargowise", "analysis")
        if not template:
            return None
            
        # Prepare default values for template variables
        template_vars = {
            'course': race_data.get('course', 'Unknown'),
            'distance': race_data.get('distance', 'Unknown'),
            'going': race_data.get('going', 'Unknown'),
            'field_size': race_data.get('field_size', 'Unknown'),
            'race_type': race_data.get('race_type', 'Unknown'),
            'race_data': self._format_race_data(race_data.get('horses', []))
        }
        
        return self.format_prompt(template, **template_vars)
    
    def get_cargowise_prediction_prompt(self, model_predictions: Dict[str, Any], 
                                      race_context: Dict[str, Any]) -> Optional[str]:
        """
        Get formatted prediction enhancement prompt using cargowise domain template
        
        Args:
            model_predictions: Dictionary containing ML model predictions
            race_context: Dictionary containing race context information
            
        Returns:
            Formatted prompt string or None
        """
        template = self.load_prompt("cargowise", "prediction")
        if not template:
            return None
            
        template_vars = {
            'model_predictions': self._format_predictions(model_predictions),
            'course': race_context.get('course', 'Unknown'),
            'race_time': race_context.get('time', 'Unknown'),
            'distance': race_context.get('distance', 'Unknown'),
            'field_size': race_context.get('field_size', 'Unknown'),
            'going': race_context.get('going', 'Unknown')
        }
        
        return self.format_prompt(template, **template_vars)
    
    def get_cargowise_strategy_prompt(self, selected_horses: list, 
                                    strategy_context: Dict[str, Any]) -> Optional[str]:
        """
        Get formatted strategy prompt using cargowise domain template
        
        Args:
            selected_horses: List of selected horses with predictions
            strategy_context: Dictionary containing strategy context
            
        Returns:
            Formatted prompt string or None
        """
        template = self.load_prompt("cargowise", "strategy")
        if not template:
            return None
            
        template_vars = {
            'selected_horses': self._format_horse_selections(selected_horses),
            'course': strategy_context.get('course', 'Unknown'),
            'race_time': strategy_context.get('time', 'Unknown'),
            'risk_profile': strategy_context.get('risk_profile', 'Moderate'),
            'capital_allocation': strategy_context.get('capital_allocation', 'Standard'),
            'market_state': strategy_context.get('market_state', 'Normal')
        }
        
        return self.format_prompt(template, **template_vars)
    
    def _format_race_data(self, horses: list) -> str:
        """Format horse data for prompt template"""
        if not horses:
            return "No horse data available"
            
        formatted_lines = []
        for horse in horses:
            line = f"- {horse.get('name', 'Unknown')}: "
            details = []
            
            if 'age' in horse:
                details.append(f"Age {horse['age']}")
            if 'weight' in horse:
                details.append(f"Weight {horse['weight']}")
            if 'jockey' in horse:
                details.append(f"Jockey: {horse['jockey']}")
            if 'trainer' in horse:
                details.append(f"Trainer: {horse['trainer']}")
            if 'recent_form' in horse:
                details.append(f"Form: {horse['recent_form']}")
                
            line += ", ".join(details)
            formatted_lines.append(line)
            
        return "\n".join(formatted_lines)
    
    def _format_predictions(self, predictions: Dict[str, Any]) -> str:
        """Format model predictions for prompt template"""
        if not predictions:
            return "No predictions available"
            
        formatted_lines = []
        for horse_name, pred_data in predictions.items():
            if isinstance(pred_data, dict):
                prob = pred_data.get('probability', 0)
                confidence = pred_data.get('confidence', 'Unknown')
                line = f"- {horse_name}: {prob:.1%} probability (confidence: {confidence})"
            else:
                line = f"- {horse_name}: {pred_data:.1%} probability"
            formatted_lines.append(line)
            
        return "\n".join(formatted_lines)
    
    def _format_horse_selections(self, horses: list) -> str:
        """Format selected horses for strategy prompt"""
        if not horses:
            return "No horses selected"
            
        formatted_lines = []
        for horse in horses:
            name = horse.get('name', 'Unknown')
            prob = horse.get('probability', 0)
            reason = horse.get('selection_reason', 'Selected by strategy')
            line = f"- {name} ({prob:.1%}): {reason}"
            formatted_lines.append(line)
            
        return "\n".join(formatted_lines)
    
    def list_available_prompts(self, domain: str = None) -> Dict[str, list]:
        """
        List all available prompts, optionally filtered by domain
        
        Args:
            domain: Optional domain filter
            
        Returns:
            Dictionary mapping domain names to lists of available prompts
        """
        available = {}
        domains_path = self.base_path / "domains"
        
        if not domains_path.exists():
            return available
            
        if domain:
            domains_to_check = [domain] if (domains_path / domain).exists() else []
        else:
            domains_to_check = [d.name for d in domains_path.iterdir() if d.is_dir()]
            
        for domain_name in domains_to_check:
            domain_path = domains_path / domain_name
            prompts = []
            
            for prompt_file in domain_path.glob("*.prompt"):
                prompts.append(prompt_file.stem)
                
            if prompts:
                available[domain_name] = sorted(prompts)
                
        return available

# Global instance for easy access
prompt_manager = PromptManager()