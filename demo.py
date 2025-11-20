from longcodezip import LongCodeZip
from loguru import logger

if __name__ == "__main__":

    with open("assets/example_context.java", "r") as f:
        context = f.read()

    question = '''
    public void switchToLoadGameScene() {
        Scene loadGameScene = new Scene(loadGameScreenController.getView(), 512, 512);

        primaryStage.setScene(loadGameScene);
        primaryStage.show();
    }
    '''
   
    # Initialize compressor
    logger.info("Initializing compressor...")
    model_name = "/Users/sckwoky/Library/Application Support/Models/Qwen2.5-Coder-0.5B"
    compressor = LongCodeZip(model_name=model_name)
    
    # Test function-based code file compression with query
    logger.info("\nTesting function-based code file compression with query...")

    original_tokens = len(compressor.tokenizer.encode(context))
    target_token = 128
    target_ratio = min(1.0, max(0.0, target_token / original_tokens))
    logger.info(f"LongCodeZip: Original tokens={original_tokens}, Target tokens={target_token}, Calculated ratio={target_ratio:.4f}")

    logger.info("\nTesting compression with Coarse-grained compression only...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=True # Coarse-grained compression
    )
    logger.info(f"Compressed prompt: \n{result_cond['compressed_prompt']}")
    logger.info(f"Compression ratio: {result_cond['compression_ratio']:.4f}") # Compression ratio: 0.3856

    logger.info("\nTesting compression with Coarse-grained and Fine-grained compression...")
    result_cond = compressor.compress_code_file(
        code=context,
        query=question,
        instruction="Complete the following code function given the context.",
        rate=target_ratio,
        rank_only=False # Corase-grained and Fine-grained compression
    )
    logger.info(f"Compressed prompt: \n{result_cond['compressed_prompt']}")
    logger.info(f"Compression ratio: {result_cond['compression_ratio']:.4f}") # Compression ratio: 0.1468