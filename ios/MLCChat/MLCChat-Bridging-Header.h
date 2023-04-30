//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//
// Exposed interface of Object-C, enables swift binding.
#import <Foundation/Foundation.h>
#include <os/proc.h>

@interface LLMChatInstance : NSObject
- (void)initialize;
- (void)evaluate;
- (void)encode:(NSString*)prompt;
- (void)decode;
- (void)reset;
- (NSString*)getMessage;
- (bool)stopped;
- (NSString*)runtimeStatsText;
@end
