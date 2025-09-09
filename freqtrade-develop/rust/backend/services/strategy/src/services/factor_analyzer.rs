use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;
use std::sync::Arc;

use crate::config::Config;
use crate::models::*;
use crate::services::signal_generator::MarketData;
use crate::indicators::*;

pub struct FactorAnalyzer {
    config: Arc<Config>,
}

impl FactorAnalyzer {
    pub async fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            config: Arc::new(config.clone()),
        })
    }

    pub async fn get_indicators(&self, symbol: &str, timeframe: &str) -> Result<IndicatorListResponse> {
        // Mock indicator calculation - in reality would fetch historical data
        let mut indicators = Vec::new();

        // RSI
        indicators.push(Indicator {
            name: "RSI".to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            value: 45.6,
            previous_value: 43.2,
            change: 2.4,
            change_pct: 5.56,
            signal: IndicatorSignal::Neutral,
            confidence: 0.85,
            calculated_at: Utc::now(),
            parameters: HashMap::from([("period".to_string(), 14.0)]),
        });

        // Moving Average (Short)
        indicators.push(Indicator {
            name: "MA20".to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            value: 42850.75,
            previous_value: 42700.50,
            change: 150.25,
            change_pct: 0.35,
            signal: IndicatorSignal::Buy,
            confidence: 0.78,
            calculated_at: Utc::now(),
            parameters: HashMap::from([("period".to_string(), 20.0)]),
        });

        // MACD
        indicators.push(Indicator {
            name: "MACD".to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            value: 125.45,
            previous_value: 98.32,
            change: 27.13,
            change_pct: 27.58,
            signal: IndicatorSignal::Buy,
            confidence: 0.72,
            calculated_at: Utc::now(),
            parameters: HashMap::from([
                ("fast_period".to_string(), 12.0),
                ("slow_period".to_string(), 26.0),
                ("signal_period".to_string(), 9.0),
            ]),
        });

        // Bollinger Bands
        indicators.extend(vec![
            Indicator {
                name: "BB_Upper".to_string(),
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                value: 44200.0,
                previous_value: 44150.0,
                change: 50.0,
                change_pct: 0.11,
                signal: IndicatorSignal::Sell,
                confidence: 0.80,
                calculated_at: Utc::now(),
                parameters: HashMap::from([
                    ("period".to_string(), 20.0),
                    ("std_dev".to_string(), 2.0),
                ]),
            },
            Indicator {
                name: "BB_Middle".to_string(),
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                value: 43000.0,
                previous_value: 42950.0,
                change: 50.0,
                change_pct: 0.12,
                signal: IndicatorSignal::Neutral,
                confidence: 0.90,
                calculated_at: Utc::now(),
                parameters: HashMap::from([
                    ("period".to_string(), 20.0),
                    ("std_dev".to_string(), 2.0),
                ]),
            },
            Indicator {
                name: "BB_Lower".to_string(),
                symbol: symbol.to_string(),
                timeframe: timeframe.to_string(),
                value: 41800.0,
                previous_value: 41750.0,
                change: 50.0,
                change_pct: 0.12,
                signal: IndicatorSignal::Buy,
                confidence: 0.80,
                calculated_at: Utc::now(),
                parameters: HashMap::from([
                    ("period".to_string(), 20.0),
                    ("std_dev".to_string(), 2.0),
                ]),
            },
        ]);

        // Stochastic
        indicators.push(Indicator {
            name: "Stochastic".to_string(),
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            value: 34.5,
            previous_value: 29.8,
            change: 4.7,
            change_pct: 15.77,
            signal: IndicatorSignal::Buy,
            confidence: 0.68,
            calculated_at: Utc::now(),
            parameters: HashMap::from([
                ("k_period".to_string(), 14.0),
                ("d_period".to_string(), 3.0),
            ]),
        });

        Ok(IndicatorListResponse {
            indicators,
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            last_updated: Utc::now(),
        })
    }

    pub async fn get_market_factors(&self) -> Result<FactorAnalysisResponse> {
        let mut factors = Vec::new();

        // Technical factors
        factors.extend(vec![
            MarketFactor {
                id: "market_momentum".to_string(),
                name: "Market Momentum".to_string(),
                description: "Overall market momentum based on price trends".to_string(),
                category: FactorCategory::Technical,
                value: 0.65,
                normalized_value: 0.3, // Z-score normalized
                percentile: 72.5,
                confidence: 0.85,
                last_updated: Utc::now(),
                data_source: "technical_analysis".to_string(),
            },
            MarketFactor {
                id: "volatility_regime".to_string(),
                name: "Volatility Regime".to_string(),
                description: "Current volatility level relative to historical norms".to_string(),
                category: FactorCategory::Technical,
                value: 0.24,
                normalized_value: -0.5,
                percentile: 35.2,
                confidence: 0.92,
                last_updated: Utc::now(),
                data_source: "volatility_analysis".to_string(),
            },
        ]);

        // Fundamental factors
        factors.extend(vec![
            MarketFactor {
                id: "btc_dominance".to_string(),
                name: "Bitcoin Dominance".to_string(),
                description: "Bitcoin's market cap as percentage of total crypto market".to_string(),
                category: FactorCategory::Fundamental,
                value: 0.54,
                normalized_value: 0.1,
                percentile: 58.3,
                confidence: 0.95,
                last_updated: Utc::now(),
                data_source: "market_cap_data".to_string(),
            },
            MarketFactor {
                id: "defi_tvl".to_string(),
                name: "DeFi Total Value Locked".to_string(),
                description: "Total value locked in DeFi protocols".to_string(),
                category: FactorCategory::Fundamental,
                value: 45.2, // Billions USD
                normalized_value: -0.2,
                percentile: 42.1,
                confidence: 0.88,
                last_updated: Utc::now(),
                data_source: "defi_protocols".to_string(),
            },
        ]);

        // Sentiment factors
        factors.extend(vec![
            MarketFactor {
                id: "fear_greed_index".to_string(),
                name: "Fear & Greed Index".to_string(),
                description: "Market sentiment indicator based on multiple metrics".to_string(),
                category: FactorCategory::Sentiment,
                value: 0.68,
                normalized_value: 0.4,
                percentile: 68.0,
                confidence: 0.79,
                last_updated: Utc::now(),
                data_source: "sentiment_analysis".to_string(),
            },
            MarketFactor {
                id: "social_media_sentiment".to_string(),
                name: "Social Media Sentiment".to_string(),
                description: "Aggregate sentiment from social media platforms".to_string(),
                category: FactorCategory::Sentiment,
                value: 0.72,
                normalized_value: 0.6,
                percentile: 74.5,
                confidence: 0.65,
                last_updated: Utc::now(),
                data_source: "social_media_analysis".to_string(),
            },
        ]);

        // Macro factors
        factors.extend(vec![
            MarketFactor {
                id: "dollar_index".to_string(),
                name: "US Dollar Index".to_string(),
                description: "Strength of US dollar relative to basket of currencies".to_string(),
                category: FactorCategory::Macro,
                value: 103.45,
                normalized_value: -0.3,
                percentile: 38.2,
                confidence: 0.98,
                last_updated: Utc::now(),
                data_source: "forex_data".to_string(),
            },
            MarketFactor {
                id: "risk_appetite".to_string(),
                name: "Global Risk Appetite".to_string(),
                description: "Market risk appetite based on cross-asset correlations".to_string(),
                category: FactorCategory::Macro,
                value: 0.58,
                normalized_value: 0.2,
                percentile: 62.8,
                confidence: 0.82,
                last_updated: Utc::now(),
                data_source: "cross_asset_analysis".to_string(),
            },
        ]);

        // On-chain factors
        factors.extend(vec![
            MarketFactor {
                id: "network_activity".to_string(),
                name: "Network Activity".to_string(),
                description: "Blockchain network activity and usage metrics".to_string(),
                category: FactorCategory::OnChain,
                value: 0.82,
                normalized_value: 0.8,
                percentile: 85.3,
                confidence: 0.91,
                last_updated: Utc::now(),
                data_source: "blockchain_metrics".to_string(),
            },
            MarketFactor {
                id: "whale_activity".to_string(),
                name: "Whale Activity".to_string(),
                description: "Large holder transaction patterns and accumulation".to_string(),
                category: FactorCategory::OnChain,
                value: 0.34,
                normalized_value: -0.4,
                percentile: 31.2,
                confidence: 0.75,
                last_updated: Utc::now(),
                data_source: "whale_tracking".to_string(),
            },
        ]);

        // Determine market regime
        let market_regime = self.determine_market_regime(&factors);
        let risk_assessment = self.assess_market_risk(&factors);
        let recommendations = self.generate_recommendations(&factors, &market_regime, &risk_assessment);

        Ok(FactorAnalysisResponse {
            factors,
            market_regime,
            risk_assessment,
            recommendations,
            calculated_at: Utc::now(),
        })
    }

    pub async fn calculate_indicators(&self, symbols: &[String], market_data: &HashMap<String, MarketData>) -> Result<HashMap<String, HashMap<String, Indicator>>> {
        let mut result = HashMap::new();

        for symbol in symbols {
            if let Some(data) = market_data.get(symbol) {
                let mut symbol_indicators = HashMap::new();

                // RSI calculation (simplified)
                let rsi_value = self.calculate_rsi(data).await?;
                symbol_indicators.insert("rsi".to_string(), Indicator {
                    name: "RSI".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: rsi_value,
                    previous_value: rsi_value - 2.5, // Mock previous value
                    change: 2.5,
                    change_pct: 5.68,
                    signal: if rsi_value < 30.0 { IndicatorSignal::StrongBuy } 
                           else if rsi_value < 40.0 { IndicatorSignal::Buy }
                           else if rsi_value > 70.0 { IndicatorSignal::StrongSell }
                           else if rsi_value > 60.0 { IndicatorSignal::Sell }
                           else { IndicatorSignal::Neutral },
                    confidence: 0.82,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 14.0)]),
                });

                // Moving averages
                let ma_short = self.calculate_moving_average(data, 20).await?;
                let ma_long = self.calculate_moving_average(data, 50).await?;

                symbol_indicators.insert("ma_short".to_string(), Indicator {
                    name: "MA20".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: ma_short,
                    previous_value: ma_short - 50.0,
                    change: 50.0,
                    change_pct: 0.12,
                    signal: IndicatorSignal::Neutral,
                    confidence: 0.90,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 20.0)]),
                });

                symbol_indicators.insert("ma_long".to_string(), Indicator {
                    name: "MA50".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: ma_long,
                    previous_value: ma_long - 75.0,
                    change: 75.0,
                    change_pct: 0.18,
                    signal: IndicatorSignal::Neutral,
                    confidence: 0.88,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 50.0)]),
                });

                // Bollinger Bands
                let (bb_upper, bb_middle, bb_lower) = self.calculate_bollinger_bands(data, 20, 2.0).await?;
                
                symbol_indicators.insert("bb_upper".to_string(), Indicator {
                    name: "BB_Upper".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: bb_upper,
                    previous_value: bb_upper - 25.0,
                    change: 25.0,
                    change_pct: 0.06,
                    signal: IndicatorSignal::Sell,
                    confidence: 0.85,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 20.0), ("std_dev".to_string(), 2.0)]),
                });

                symbol_indicators.insert("bb_middle".to_string(), Indicator {
                    name: "BB_Middle".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: bb_middle,
                    previous_value: bb_middle - 30.0,
                    change: 30.0,
                    change_pct: 0.07,
                    signal: IndicatorSignal::Neutral,
                    confidence: 0.90,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 20.0), ("std_dev".to_string(), 2.0)]),
                });

                symbol_indicators.insert("bb_lower".to_string(), Indicator {
                    name: "BB_Lower".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: bb_lower,
                    previous_value: bb_lower - 35.0,
                    change: 35.0,
                    change_pct: 0.08,
                    signal: IndicatorSignal::Buy,
                    confidence: 0.85,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([("period".to_string(), 20.0), ("std_dev".to_string(), 2.0)]),
                });

                // MACD
                let macd_value = self.calculate_macd(data).await?;
                symbol_indicators.insert("macd".to_string(), Indicator {
                    name: "MACD".to_string(),
                    symbol: symbol.clone(),
                    timeframe: "1h".to_string(),
                    value: macd_value,
                    previous_value: macd_value - 15.5,
                    change: 15.5,
                    change_pct: 18.3,
                    signal: if macd_value > 0.0 { IndicatorSignal::Buy } else { IndicatorSignal::Sell },
                    confidence: 0.76,
                    calculated_at: Utc::now(),
                    parameters: HashMap::from([
                        ("fast_period".to_string(), 12.0),
                        ("slow_period".to_string(), 26.0),
                        ("signal_period".to_string(), 9.0),
                    ]),
                });

                result.insert(symbol.clone(), symbol_indicators);
            }
        }

        Ok(result)
    }

    pub async fn calculate_factors(&self, market_data: &HashMap<String, MarketData>) -> Result<Vec<MarketFactor>> {
        let mut factors = Vec::new();

        // Market momentum factor based on price trends
        let momentum = self.calculate_market_momentum(market_data).await?;
        factors.push(MarketFactor {
            id: "market_momentum".to_string(),
            name: "Market Momentum".to_string(),
            description: "Aggregate momentum across all tracked symbols".to_string(),
            category: FactorCategory::Technical,
            value: momentum,
            normalized_value: (momentum - 0.5) * 2.0, // Normalize to -1 to 1
            percentile: momentum * 100.0,
            confidence: 0.8,
            last_updated: Utc::now(),
            data_source: "live_market_data".to_string(),
        });

        // Volatility factor
        let volatility = self.calculate_market_volatility(market_data).await?;
        factors.push(MarketFactor {
            id: "market_volatility".to_string(),
            name: "Market Volatility".to_string(),
            description: "Current volatility level across all symbols".to_string(),
            category: FactorCategory::Technical,
            value: volatility,
            normalized_value: (volatility - 0.3) / 0.2, // Assuming 30% is normal, normalize around that
            percentile: volatility * 100.0,
            confidence: 0.85,
            last_updated: Utc::now(),
            data_source: "live_market_data".to_string(),
        });

        // Volume factor
        let volume_strength = self.calculate_volume_strength(market_data).await?;
        factors.push(MarketFactor {
            id: "volume_strength".to_string(),
            name: "Volume Strength".to_string(),
            description: "Current volume levels relative to averages".to_string(),
            category: FactorCategory::Technical,
            value: volume_strength,
            normalized_value: (volume_strength - 1.0) / 0.5, // Normalize around 1.0 (normal volume)
            percentile: (volume_strength * 50.0).min(100.0),
            confidence: 0.75,
            last_updated: Utc::now(),
            data_source: "live_market_data".to_string(),
        });

        Ok(factors)
    }

    // Private calculation methods
    async fn calculate_rsi(&self, data: &MarketData) -> Result<f64> {
        // Simplified RSI calculation - in reality would need historical data
        let current_price = data.close_price.to_f64().unwrap_or(0.0);
        let open_price = data.open_price.to_f64().unwrap_or(current_price);
        
        if current_price > open_price {
            Ok(65.0) // Bullish RSI
        } else {
            Ok(35.0) // Bearish RSI
        }
    }

    async fn calculate_moving_average(&self, data: &MarketData, _period: u32) -> Result<f64> {
        // Simplified MA - in reality would calculate from historical prices
        let current_price = data.close_price.to_f64().unwrap_or(0.0);
        Ok(current_price * 0.99) // Slightly below current price
    }

    async fn calculate_bollinger_bands(&self, data: &MarketData, _period: u32, std_dev: f64) -> Result<(f64, f64, f64)> {
        // Simplified Bollinger Bands calculation
        let current_price = data.close_price.to_f64().unwrap_or(0.0);
        let middle = current_price * 0.99;
        let band_width = current_price * 0.02 * std_dev; // 2% * std_dev
        
        Ok((
            middle + band_width, // Upper
            middle,             // Middle
            middle - band_width, // Lower
        ))
    }

    async fn calculate_macd(&self, data: &MarketData) -> Result<f64> {
        // Simplified MACD calculation
        let current_price = data.close_price.to_f64().unwrap_or(0.0);
        let open_price = data.open_price.to_f64().unwrap_or(current_price);
        
        // Return momentum-based MACD approximation
        (current_price - open_price) / open_price * 1000.0
    }

    async fn calculate_market_momentum(&self, market_data: &HashMap<String, MarketData>) -> Result<f64> {
        let mut total_momentum = 0.0;
        let mut count = 0;

        for data in market_data.values() {
            let current = data.close_price.to_f64().unwrap_or(0.0);
            let open = data.open_price.to_f64().unwrap_or(current);
            
            if open > 0.0 {
                total_momentum += (current - open) / open;
                count += 1;
            }
        }

        if count > 0 {
            let avg_momentum = total_momentum / count as f64;
            Ok((avg_momentum + 1.0) / 2.0) // Normalize to 0-1 range
        } else {
            Ok(0.5) // Neutral
        }
    }

    async fn calculate_market_volatility(&self, market_data: &HashMap<String, MarketData>) -> Result<f64> {
        let mut total_volatility = 0.0;
        let mut count = 0;

        for data in market_data.values() {
            let high = data.high_price.to_f64().unwrap_or(0.0);
            let low = data.low_price.to_f64().unwrap_or(0.0);
            let close = data.close_price.to_f64().unwrap_or(0.0);
            
            if close > 0.0 {
                let daily_range = (high - low) / close;
                total_volatility += daily_range;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_volatility / count as f64)
        } else {
            Ok(0.02) // Default 2% volatility
        }
    }

    async fn calculate_volume_strength(&self, market_data: &HashMap<String, MarketData>) -> Result<f64> {
        let mut total_volume_ratio = 0.0;
        let mut count = 0;

        for data in market_data.values() {
            if data.average_volume > 0.0 {
                total_volume_ratio += data.volume / data.average_volume;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_volume_ratio / count as f64)
        } else {
            Ok(1.0) // Normal volume
        }
    }

    fn determine_market_regime(&self, factors: &[MarketFactor]) -> MarketRegime {
        let momentum = factors.iter()
            .find(|f| f.id == "market_momentum")
            .map(|f| f.normalized_value)
            .unwrap_or(0.0);

        let volatility = factors.iter()
            .find(|f| f.category == FactorCategory::Technical && f.name.contains("Volatility"))
            .map(|f| f.normalized_value)
            .unwrap_or(0.0);

        let sentiment = factors.iter()
            .filter(|f| f.category == FactorCategory::Sentiment)
            .map(|f| f.normalized_value)
            .sum::<f64>() / factors.iter().filter(|f| f.category == FactorCategory::Sentiment).count().max(1) as f64;

        let regime = if momentum > 0.5 && sentiment > 0.3 {
            "Bull"
        } else if momentum < -0.3 && sentiment < -0.2 {
            "Bear"
        } else if volatility > 0.5 {
            "Volatile"
        } else {
            "Sideways"
        };

        let confidence = (momentum.abs() + volatility.abs() + sentiment.abs()) / 3.0;

        MarketRegime {
            regime: regime.to_string(),
            confidence,
            duration_days: 14, // Mock duration
            characteristics: match regime {
                "Bull" => vec!["Rising prices".to_string(), "High sentiment".to_string(), "Strong momentum".to_string()],
                "Bear" => vec!["Falling prices".to_string(), "Low sentiment".to_string(), "Negative momentum".to_string()],
                "Volatile" => vec!["High volatility".to_string(), "Uncertain direction".to_string(), "Increased risk".to_string()],
                _ => vec!["Range-bound".to_string(), "Low volatility".to_string(), "Consolidation".to_string()],
            },
        }
    }

    fn assess_market_risk(&self, factors: &[MarketFactor]) -> RiskAssessment {
        let volatility_risk = factors.iter()
            .find(|f| f.name.contains("Volatility"))
            .map(|f| (f.normalized_value + 1.0) * 50.0)
            .unwrap_or(50.0);

        let market_risk = factors.iter()
            .filter(|f| f.category == FactorCategory::Macro)
            .map(|f| f.normalized_value.abs() * 100.0)
            .sum::<f64>() / factors.iter().filter(|f| f.category == FactorCategory::Macro).count().max(1) as f64;

        let liquidity_risk = 30.0; // Mock value
        let correlation_risk = 40.0; // Mock value

        let overall_risk = (volatility_risk + market_risk + liquidity_risk + correlation_risk) / 4.0;

        let mut recommendations = Vec::new();
        if overall_risk > 70.0 {
            recommendations.push("Consider reducing position sizes".to_string());
            recommendations.push("Implement tighter stop losses".to_string());
        } else if overall_risk < 30.0 {
            recommendations.push("Low risk environment - consider increasing exposure".to_string());
        }

        RiskAssessment {
            overall_risk,
            market_risk,
            liquidity_risk,
            volatility_risk,
            correlation_risk,
            recommendations,
        }
    }

    fn generate_recommendations(&self, factors: &[MarketFactor], market_regime: &MarketRegime, risk_assessment: &RiskAssessment) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Regime-based recommendations
        match market_regime.regime.as_str() {
            "Bull" => {
                recommendations.push("Consider momentum strategies".to_string());
                recommendations.push("Focus on breakout patterns".to_string());
                if risk_assessment.overall_risk < 50.0 {
                    recommendations.push("Consider increasing position sizes".to_string());
                }
            },
            "Bear" => {
                recommendations.push("Consider mean reversion strategies".to_string());
                recommendations.push("Implement protective stops".to_string());
                recommendations.push("Consider short positions or hedging".to_string());
            },
            "Volatile" => {
                recommendations.push("Use smaller position sizes".to_string());
                recommendations.push("Consider range trading strategies".to_string());
                recommendations.push("Implement tighter risk management".to_string());
            },
            _ => {
                recommendations.push("Consider range-bound strategies".to_string());
                recommendations.push("Focus on mean reversion patterns".to_string());
            },
        }

        // Risk-based recommendations
        if risk_assessment.volatility_risk > 70.0 {
            recommendations.push("High volatility detected - reduce leverage".to_string());
        }

        if risk_assessment.market_risk > 60.0 {
            recommendations.push("Elevated market risk - diversify strategies".to_string());
        }

        // Factor-based recommendations
        let sentiment_avg = factors.iter()
            .filter(|f| f.category == FactorCategory::Sentiment)
            .map(|f| f.normalized_value)
            .sum::<f64>() / factors.iter().filter(|f| f.category == FactorCategory::Sentiment).count().max(1) as f64;

        if sentiment_avg > 0.7 {
            recommendations.push("Extreme bullish sentiment - watch for reversals".to_string());
        } else if sentiment_avg < -0.7 {
            recommendations.push("Extreme bearish sentiment - potential contrarian opportunity".to_string());
        }

        recommendations
    }
}