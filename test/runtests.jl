using SafeTestsets

# Use all tests or just the active one
use_all_tests = false
active_test = "Default OptMap test"

@time begin
    # Run all of the tests
    if use_all_tests
        println("Running all tests ...")

    # Only run the active test
    else
        println("Only running active test ($active_test) ...")
        @time @safetestset "Default OptMap test" begin
            include("test_nloptmap.jl")
        end
    end

end
