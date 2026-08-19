/* H1 venue-parity fuzz driver (verification artifact, not the deliverable).
 *
 * Runs the OFFICIAL CodinGame UTTT referee (github.com/CodinGame/game-ultimate-tictactoe,
 * built locally with its own SDK) headlessly, with napkin_referee.py's CG protocol
 * adapter playing BOTH seats on a random policy. The adapter recomputes the valid-action
 * set every turn and prints PARITY MISMATCH to stderr on any divergence from what the
 * referee sent; when its own move ends the game it prints its predicted final scores
 * (PREDICT line). This driver surfaces both, plus the referee's real scores, per game.
 *
 * Run from the referee repo (see README "Repro"):
 *   javac -cp target/classes:<runner deps> -d target/fuzz FuzzMain.java
 *   java  -cp target/fuzz:target/classes:<runner deps> FuzzMain <games> <level> <seed0>
 * Output: one "GAME <seed> scores={0=s0, 1=s1} predict=<agent>:PREDICT ..." line per
 * game (+ any PARITY lines), then "TOTAL parity_mismatch_lines=N". The README's Repro
 * block asserts predict==scores for every game and N==0.
 */
import com.codingame.gameengine.runner.MultiplayerGameRunner;
import com.codingame.gameengine.runner.dto.GameResult;

import java.util.List;
import java.util.Map;

public class FuzzMain {
    public static void main(String[] args) throws Exception {
        int games = Integer.parseInt(args[0]);
        int level = Integer.parseInt(args[1]);
        long seed0 = Long.parseLong(args[2]);
        String py = System.getenv().getOrDefault("NAPKIN_REFEREE_PY",
                System.getProperty("user.dir") + "/napkin_referee.py");
        // optional custom agent commands (args 3/4); "SEED" is replaced per game
        String agent0 = args.length > 3 ? args[3]
                : "python3 " + py + " cg --policy random --level " + level + " --seed SEED0";
        String agent1 = args.length > 4 ? args[4]
                : "python3 " + py + " cg --policy random --level " + level + " --seed SEED1";
        int parityLines = 0;

        for (int g = 0; g < games; g++) {
            long seed = seed0 + g;
            System.setProperty("league.level", Integer.toString(level));
            MultiplayerGameRunner runner = new MultiplayerGameRunner();
            runner.setSeed(seed);
            runner.addAgent(agent0.replace("SEED0", Long.toString(seed * 2)));
            runner.addAgent(agent1.replace("SEED1", Long.toString(seed * 2 + 1)));
            GameResult res = runner.simulate();

            String predict = null;
            for (Map.Entry<String, List<String>> e : res.errors.entrySet()) {
                for (String chunk : e.getValue()) {
                    if (chunk == null) continue;
                    for (String line : chunk.split("\n")) {
                        if (line.startsWith("PREDICT")) {
                            predict = e.getKey() + ":" + line;
                        } else if (line.contains("PARITY")) {
                            parityLines++;
                            System.out.println("GAME " + seed + " " + line);
                        }
                    }
                }
            }
            System.out.println("GAME " + seed + " level=" + level
                    + " scores=" + res.scores + " predict=" + predict);
        }
        System.out.println("TOTAL parity_mismatch_lines=" + parityLines);
    }
}
