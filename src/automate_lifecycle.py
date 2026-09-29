import mlflow
from mlflow.tracking import MlflowClient

def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")
    
    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"
    metric_to_optimize = "recall" # We use Recall as our decision metric
    
    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        # 1. Discover untracked models (Stage: None) and push them to Staging
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]
        if new_models:
            print(f"\n[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")
            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"  -> Version {mv.version} is now in Staging.")
                
        # Refresh registry data
        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]
        
        if not staging_models:
            print("\n[INFO] No models in Staging to evaluate. Exiting.")
            return
            
        # 2. Evaluate Staging Models (Find the Best Challenger)
        print("\n[INFO] Evaluating Staging candidates...")
        best_challenger = None
        best_challenger_score = -1.0
        
        for mv in staging_models:
            run = client.get_run(mv.run_id)
            score = run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"  -> Staging Candidate: Version {mv.version} | {metric_to_optimize}: {score:.4f}")
            
            if score > best_challenger_score:
                best_challenger_score = score
                best_challenger = mv
                
        print(f"\n[INFO] Best Challenger is Version {best_challenger.version} ({metric_to_optimize}: {best_challenger_score:.4f})")
        
        # 3. Find the Current Champion in Production
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None
        promote_challenger = False
        
        if not current_champion:
            print("[INFO] No model currently in Production. Challenger wins by default!")
            promote_challenger = True
        else:
            champ_run = client.get_run(current_champion.run_id)
            champ_score = champ_run.data.metrics.get(metric_to_optimize, 0.0)
            print(f"[INFO] Current Production Champion: Version {current_champion.version} | {metric_to_optimize}: {champ_score:.4f}")
            
            # 4. The Face-Off
            if best_challenger_score > champ_score:
                print("[SUCCESS] Challenger beat the Champion!")
                promote_challenger = True
            else:
                print("[INFO] Challenger failed to beat the Champion. Champion retains its title.")
                
        # 5. Execute Lifecycle Promotions & Rollbacks
        if promote_challenger:
            print(f"\n[INFO] Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            if current_champion:
                print(f"[INFO] Archiving defeated Champion (Version {current_champion.version})...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )
                
        # 6. Clean up Staging (Archive all remaining losers)
        # Refresh registry one last time to see what was left behind in Staging
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging = [mv for mv in final_versions if mv.current_stage == "Staging"]
        
        if remaining_staging:
            print("\n[INFO] Cleaning up Staging area...")
            for mv in remaining_staging:
                print(f"  -> Archiving Version {mv.version} (Lost the competition)")
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
                )
                
        print("\n[SUCCESS] Automated Model Lifecycle execution complete!")
        
    except Exception as e:
        print(f"[ERROR] Failed to automate lifecycle: {e}")

if __name__ == "__main__":
    automate_champion_challenger()