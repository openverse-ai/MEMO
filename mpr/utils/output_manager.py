from pathlib import Path
from dataclasses import dataclass


@dataclass
class OutputManager:
    project_root: Path
    trajectories_dir: Path
    summaries_dir: Path
    prompts_dir: Path
    memory_dir: Path
    temp_dir: Path

    @classmethod
    def init(cls, project_name: Path) -> "OutputManager":
        project_root = (Path("logs") / project_name).absolute()
        project_root.mkdir(parents=True, exist_ok=True)

        trajectories_dir = project_root / "trajectories"
        summaries_dir = project_root / "summaries"
        prompts_dir = project_root / "prompts"
        memory_dir = project_root / "memory"
        temp_dir = project_root / "temp"

        for d in [
            trajectories_dir,
            summaries_dir,
            prompts_dir,
            memory_dir,
            temp_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        return cls(
            project_root=project_root,
            trajectories_dir=trajectories_dir,
            summaries_dir=summaries_dir,
            prompts_dir=prompts_dir,
            memory_dir=memory_dir,
            temp_dir=temp_dir,
        )

    def organize_tournament_files(
        self,
        temp_dir: Path,
        generation: int,
        evaluation_phase: bool = False,
    ):
        """Move tournament files to organized directories."""

        if evaluation_phase:
            # Create evaluation trajectories directory if needed
            eval_traj_dir = self.project_root / "evaluation_trajectories"
            eval_traj_dir.mkdir(exist_ok=True)

            # Move trajectory files for evaluation
            for traj_file in temp_dir.glob("trajectories_*.json"):
                dest = eval_traj_dir / f"gen{generation}_eval_{traj_file.name}"
                traj_file.rename(dest)
        else:
            # Move trajectory files for evolution
            for traj_file in temp_dir.glob("trajectories_*.json"):
                dest = self.trajectories_dir / f"gen{generation}_{traj_file.name}"
                traj_file.rename(dest)

        # Move summary files
        for summary_file in temp_dir.glob("*.json"):
            if not summary_file.name.startswith("trajectories_"):
                if evaluation_phase:
                    dest = self.summaries_dir / f"gen{generation}_eval_{summary_file.name}"
                else:
                    dest = self.summaries_dir / f"gen{generation}_{summary_file.name}"
                summary_file.rename(dest)

        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except OSError:
            pass
