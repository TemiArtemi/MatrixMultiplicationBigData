import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;

/**
 * Matrix multiplication benchmark - Java implementation.
 * Implements basic O(n³) triple-loop matrix multiplication.
 * Reads matrices from CSV, measures kernel time and memory.
 */
public class MatrixMult {
    
    // Configuration
    private static final int SEED = 42;
    private static final double VALUE_MIN = -10.0;
    private static final double VALUE_MAX = 10.0;
    private static final int[] SIZES = {10, 50, 100, 200, 400, 800, 1000, 1500, 2000, 3000, 4000};
    private static final String MATRICES_DIR = "data/matrices";
    private static final String RESULTS_DIR = "data/results";
    
    // Benchmark settings
    private static final int WARMUP_RUNS = 5;
    private static final int MIN_REPETITIONS = 5;
    private static final long TIME_BUDGET_NS = 180_000_000_000L; // 3 minutes in nanoseconds
    
    // Batch sizes for very small matrices (size -> batches)
    private static final Map<Integer, Integer> BATCH_SIZES = Map.of(
        10, 1000,
        50, 100,
        100, 10
    );
    
    // Tolerances for correctness
    private static final double ABS_TOL = 1e-9;
    private static final double REL_TOL = 1e-9;
    
    private static final MemoryMXBean MEMORY_BEAN = ManagementFactory.getMemoryMXBean();
    
    public static void main(String[] args) throws Exception {
        boolean validateOnly = false;
        List<Integer> sizesToTest = new ArrayList<>();
        
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--validate-only")) {
                validateOnly = true;
            } else if (args[i].equals("--sizes")) {
                while (i + 1 < args.length && !args[i + 1].startsWith("--")) {
                    sizesToTest.add(Integer.parseInt(args[++i]));
                }
            }
        }
        
        Path matricesPath = Paths.get(MATRICES_DIR);
        Path resultsPath = Paths.get(RESULTS_DIR);
        Files.createDirectories(resultsPath);
        
        System.out.println("Java Matrix Multiplication Benchmark");
        System.out.println("Matrices directory: " + matricesPath.toAbsolutePath());
        System.out.println("Results directory: " + resultsPath.toAbsolutePath());
        System.out.println("Sizes: " + Arrays.toString(SIZES));
        System.out.println("Warm-up runs: " + WARMUP_RUNS);
        System.out.println("Min repetitions: " + MIN_REPETITIONS);
        System.out.println("Time budget: " + (TIME_BUDGET_NS / 1_000_000_000.0) + "s per size");
        System.out.println("Java version: " + System.getProperty("java.version"));
        System.out.println("Available processors: " + Runtime.getRuntime().availableProcessors());
        
        // Run correctness validation
        if (!validateCorrectness()) {
            System.err.println("Correctness validation failed. Aborting benchmark.");
            System.exit(1);
        }
        
        if (validateOnly) {
            return;
        }
        
        // Run benchmarks
        int[] sizes = sizesToTest.isEmpty() ? SIZES : sizesToTest.stream().mapToInt(Integer::intValue).toArray();
        List<Result> allResults = new ArrayList<>();
        
        for (int n : sizes) {
            List<Result> results = benchmarkSize(n);
            allResults.addAll(results);
        }
        
        // Save raw results to CSV
        if (!allResults.isEmpty()) {
            String timestamp = new java.text.SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            Path outputFile = resultsPath.resolve("results_java_" + timestamp + ".csv");
            
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile))) {
                writer.println("language,size,numeric_type,repetition,time_ms,memory_mb");
                for (Result r : allResults) {
                    writer.printf("%s,%d,%s,%d,%.2f,%.2f%n",
                        r.language, r.size, r.numericType, r.repetition, r.timeMs, r.memoryMb);
                }
            }
            
            System.out.println("\nResults saved to: " + outputFile);
            
            // Print summary
            System.out.println("\nSUMMARY (median time per size):");
            System.out.println("-".repeat(50));
            for (int n : sizes) {
                List<Result> sizeResults = allResults.stream()
                    .filter(r -> r.size == n)
                    .toList();
                if (!sizeResults.isEmpty()) {
                    double[] times = sizeResults.stream().mapToDouble(r -> r.timeMs).sorted().toArray();
                    double median = times[times.length / 2];
                    double q1 = times[times.length / 4];
                    double q3 = times[3 * times.length / 4];
                    double iqr = q3 - q1;
                    System.out.printf("  n=%5d: %10.2f ms (IQR: %.2f ms)%n", n, median, iqr);
                }
            }
        }
        
        System.out.println("\nBenchmark complete.");
    }
    
    /**
     * Load matrix from CSV file.
     */
    private static double[][] loadMatrixCSV(Path filepath) throws IOException {
        List<String> lines = Files.readAllLines(filepath);
        double[][] matrix = new double[lines.size()][];
        for (int i = 0; i < lines.size(); i++) {
            String[] parts = lines.get(i).split(",");
            matrix[i] = new double[parts.length];
            for (int j = 0; j < parts.length; j++) {
                matrix[i][j] = Double.parseDouble(parts[j]);
            }
        }
        return matrix;
    }
    
    /**
     * Basic triple-loop matrix multiplication: C = A × B.
     */
    private static double[][] multiply(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double s = 0.0;
                for (int k = 0; k < n; k++) {
                    s += A[i][k] * B[k][j];
                }
                C[i][j] = s;
            }
        }
        return C;
    }
    
    /**
     * Run multiplication multiple times (for very small matrices) and return last result.
     */
    private static double[][] multiplyBatch(double[][] A, double[][] B, int batches) {
        double[][] C = null;
        for (int b = 0; b < batches; b++) {
            C = multiply(A, B);
        }
        return C;
    }
    
    /**
     * Check if two matrices are equal within tolerances.
     */
    private static boolean matricesEqual(double[][] A, double[][] B, double absTol, double relTol) {
        if (A.length != B.length) return false;
        for (int i = 0; i < A.length; i++) {
            if (A[i].length != B[i].length) return false;
            for (int j = 0; j < A[i].length; j++) {
                double a = A[i][j], b = B[i][j];
                double diff = Math.abs(a - b);
                if (diff > absTol && diff > relTol * Math.max(Math.abs(a), Math.abs(b))) {
                    return false;
                }
            }
        }
        return true;
    }
    
    /**
     * Create n×n identity matrix.
     */
    private static double[][] identityMatrix(int n) {
        double[][] I = new double[n][n];
        for (int i = 0; i < n; i++) {
            I[i][i] = 1.0;
        }
        return I;
    }
    
    /**
     * Create n×n zero matrix.
     */
    private static double[][] zeroMatrix(int n) {
        return new double[n][n];
    }
    
    /**
     * Run correctness tests.
     */
    private static boolean validateCorrectness() {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("CORRECTNESS VALIDATION");
        System.out.println("=".repeat(60));
        
        boolean allPassed = true;
        Random rng = new Random(SEED);
        
        // Test 1: Identity matrix (A × I = A)
        System.out.println("\n1. Testing with identity matrix...");
        int n = 5;
        double[][] A = generateRandomMatrix(n, rng);
        double[][] I = identityMatrix(n);
        double[][] C = multiply(A, I);
        if (matricesEqual(C, A, ABS_TOL, REL_TOL)) {
            System.out.println("   PASS: A × I = A");
        } else {
            System.out.println("   FAIL: A × I != A");
            allPassed = false;
        }
        
        // Test 2: Zero matrix (A × 0 = 0)
        System.out.println("\n2. Testing with zero matrix...");
        double[][] Z = zeroMatrix(n);
        C = multiply(A, Z);
        double[][] expected = zeroMatrix(n);
        if (matricesEqual(C, expected, ABS_TOL, REL_TOL)) {
            System.out.println("   PASS: A × 0 = 0");
        } else {
            System.out.println("   FAIL: A × 0 != 0");
            allPassed = false;
        }
        
        // Test 3: Small known matrices
        System.out.println("\n3. Testing with known small matrices...");
        double[][] A_test = {{1.0, 2.0}, {3.0, 4.0}};
        double[][] B_test = {{5.0, 6.0}, {7.0, 8.0}};
        double[][] expected_test = {{19.0, 22.0}, {43.0, 50.0}};
        C = multiply(A_test, B_test);
        if (matricesEqual(C, expected_test, ABS_TOL, REL_TOL)) {
            System.out.println("   PASS: 2×2 known result");
        } else {
            System.out.println("   FAIL: 2×2 known result differs");
            allPassed = false;
        }
        
        // Test 4: Reference comparison with a simple reference implementation
        System.out.println("\n4. Testing against reference implementation...");
        n = 20;
        A = generateRandomMatrix(n, rng);
        double[][] B = generateRandomMatrix(n, rng);
        C = multiply(A, B);
        
        // Reference: different loop order (k, i, j) - should give same mathematical result
        double[][] C_ref = multiplyReference(A, B);
        if (matricesEqual(C, C_ref, ABS_TOL, REL_TOL)) {
            System.out.println("   PASS: Matches reference implementation");
        } else {
            System.out.println("   FAIL: Differs from reference implementation");
            allPassed = false;
        }
        
        System.out.println("\n" + "=".repeat(60));
        if (allPassed) {
            System.out.println("ALL CORRECTNESS TESTS PASSED");
        } else {
            System.out.println("SOME CORRECTNESS TESTS FAILED");
        }
        System.out.println("=".repeat(60));
        
        return allPassed;
    }
    
    /**
     * Generate random matrix with given seed.
     */
    private static double[][] generateRandomMatrix(int n, Random rng) {
        double[][] matrix = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                matrix[i][j] = VALUE_MIN + rng.nextDouble() * (VALUE_MAX - VALUE_MIN);
            }
        }
        return matrix;
    }
    
    /**
     * Reference implementation with different loop order (k, i, j).
     */
    private static double[][] multiplyReference(double[][] A, double[][] B) {
        int n = A.length;
        double[][] C = new double[n][n];
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                double aik = A[i][k];
                for (int j = 0; j < n; j++) {
                    C[i][j] += aik * B[k][j];
                }
            }
        }
        return C;
    }
    
    /**
     * Get current heap memory usage in MB.
     */
    private static double getMemoryMB() {
        MemoryUsage heapUsage = MEMORY_BEAN.getHeapMemoryUsage();
        return heapUsage.getUsed() / (1024.0 * 1024.0);
    }
    
    /**
     * Benchmark a specific matrix size.
     */
    private static List<Result> benchmarkSize(int n) {
        System.out.println("\n  Benchmarking n=" + n + "...");
        
        Path aPath = Paths.get(MATRICES_DIR, "A_" + n + ".csv");
        Path bPath = Paths.get(MATRICES_DIR, "B_" + n + ".csv");
        
        if (!Files.exists(aPath) || !Files.exists(bPath)) {
            System.out.println("    Matrices not found for n=" + n + ", skipping");
            return List.of();
        }
        
        try {
            double[][] A = loadMatrixCSV(aPath);
            double[][] B = loadMatrixCSV(bPath);
            
            // Pre-allocate result matrix (outside timing)
            double[][] C = new double[n][n];
            
            // Warm-up runs
            System.out.println("    Warm-up (" + WARMUP_RUNS + " runs)...");
            for (int i = 0; i < WARMUP_RUNS; i++) {
                multiply(A, B);
            }
            
            // Force GC before timed runs
            System.gc();
            try { Thread.sleep(100); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            
            // Determine if we need batching
            int batches = BATCH_SIZES.getOrDefault(n, 1);
            
            // Timed runs
            List<Result> results = new ArrayList<>();
            for (int rep = 0; rep < MIN_REPETITIONS; rep++) {
                // Measure memory before
                double memBefore = getMemoryMB();
                
                // Time the kernel
                long start = System.nanoTime();
                if (batches > 1) {
                    multiplyBatch(A, B, batches);
                } else {
                    multiply(A, B);
                }
                long end = System.nanoTime();
                
                // Measure memory after
                double memAfter = getMemoryMB();
                
                double elapsedMs = (end - start) / 1_000_000.0;
                if (batches > 1) {
                    elapsedMs /= batches;
                }
                
                double memoryMb = Math.max(memAfter - memBefore, 0);
                
                Result result = new Result("Java", n, "float64", rep + 1, elapsedMs, memoryMb);
                results.add(result);
                
                System.out.printf("    Rep %d: %.2f ms, memory delta: %.2f MB%n", 
                    rep + 1, elapsedMs, memoryMb);
                
                // Check time budget
                if ((end - start) > TIME_BUDGET_NS) {
                    System.out.println("    Time budget exceeded, stopping");
                    break;
                }
            }
            
            return results;
            
        } catch (IOException | OutOfMemoryError e) {
            System.out.println("    Error for n=" + n + ": " + e.getMessage());
            return List.of();
        }
    }
    
    /**
     * Result data class.
     */
    private static class Result {
        final String language;
        final int size;
        final String numericType;
        final int repetition;
        final double timeMs;
        final double memoryMb;
        
        Result(String language, int size, String numericType, int repetition, double timeMs, double memoryMb) {
            this.language = language;
            this.size = size;
            this.numericType = numericType;
            this.repetition = repetition;
            this.timeMs = timeMs;
            this.memoryMb = memoryMb;
        }
    }
}