#ifndef BAO_PLANNER_H
#define BAO_PLANNER_H

#include <unistd.h>
#include "bao_configs.h"
#include "bao_util.h"
#include "bao_bufferstate.h"


// Functions to help with Bao query planning.


// This macro can be used to wrap a codeblock to save and restore the current
// values of the plan hints.
#define save_arm_options(x) { \
  bool hj = enable_hashjoin;\
  bool mj = enable_mergejoin;\
  bool nl = enable_nestloop;\
  bool is = enable_indexscan;\
  bool ss = enable_seqscan;\
  bool io = enable_indexonlyscan;\
  { x } \
  enable_hashjoin = hj;\
  enable_mergejoin = mj;\
  enable_nestloop = nl;\
  enable_indexscan = is;\
  enable_seqscan = ss;\
  enable_indexonlyscan = io; }




// Connect to a Bao server, construct plans for each arm, have the server
// select a plan. Has the same signature as the PG optimizer.
BaoPlan *plan_query(Query *parse, int cursorOptions, ParamListInfo boundParams);

// Translate an arm index into SQL statements to give the hint (used for EXPLAIN).
char* arm_to_hint(int arm);



// Set the planner hint options to the correct one for the passed-in arm
// index. Should be called with the `save_arm_options` macro so we don't
// blast-away the user's config.
// static void set_arm_options(int arm) {
//   enable_hashjoin = false;
//   enable_mergejoin = false;
//   enable_nestloop = false;
//   enable_indexscan = false;
//   enable_seqscan = false;
//   enable_indexonlyscan = false;
  
//   switch (arm) {
//   case 0:
//     enable_hashjoin = true;
//     enable_indexscan = true;
//     enable_mergejoin = true;
//     enable_nestloop = true;
//     enable_seqscan = true;
//     enable_indexonlyscan = true;
//     break;
    
//   case 1: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     enable_seqscan = true; 
//     break;
//   case 2: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 3: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_seqscan = true; 
//     break;
//   case 4: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 5: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     break;
//   case 6:
//     enable_hashjoin = true; 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     break;
//   case 7: 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     break;
//   case 8: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     break;
//   case 9: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     break;
//   case 10:
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_seqscan = true; 
//     break;
//   case 11: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 12: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     enable_seqscan = true; 
//     break;
//   case 13: 
//     enable_hashjoin = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     break;
//   case 14: 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     break;
//   case 15: 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 16: 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     break;
//   case 17: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     break;
//   case 18: 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     break;
//   case 19: 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 20: 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 21: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_indexscan = true; 
//     enable_mergejoin = true; 
//     break;
//   case 22: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_mergejoin = true; 
//     break;
//   case 23: 
//     enable_hashjoin = true; 
//     enable_indexscan = true; 
//     enable_nestloop = true; 
//     enable_seqscan = true; 
//     break;
//   case 24: 
//     enable_hashjoin = true; 
//     enable_indexscan = true; 
//     break;
//   case 25: 
//     enable_hashjoin = true; 
//     enable_indexonlyscan = true; 
//     enable_nestloop = true; 
//     break;
//   default:
//     elog(ERROR, "Invalid arm index %d selected.", arm);
//     break;
//   }
// }


// // Get a query plan for a particular arm.
// static PlannedStmt* plan_arm(int arm, Query* parse,
//                              int cursorOptions, ParamListInfo boundParams) {

//   PlannedStmt* plan = NULL;
//   Query* query_copy = copyObject(parse); // create a copy of the query plan

//   if (arm == -1) {
//     // Use whatever the user has set as the current configuration.
//     plan = standard_planner(query_copy, cursorOptions, boundParams);
//     return plan;
//   }
  
//   // Preserving the user's options, set the config to match the arm index
//   // and invoke the PG planner.
//   save_arm_options({
//       set_arm_options(arm);
//       plan = standard_planner(query_copy, cursorOptions, boundParams);
//     });

//   return plan;
// }
static void
set_arm_options(int arm)
{
    /* First disable everything */
    enable_hashjoin = false;
    enable_indexonlyscan = false;
    enable_indexscan = false;
    enable_mergejoin = false;
    enable_nestloop = false;
    enable_seqscan = false;

    /*
     * We have 49 valid arms (0..48). Each arm number corresponds exactly
     * to an entry in the Python _ARMS list. The mapping below follows that
     * list literally.
     */
    switch (arm)
    {
        /* 0: ["hashjoin", "indexonlyscan", "indexscan", "mergejoin", "nestloop", "seqscan"] */
        case 0:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;
        

        /*  1: ["hashjoin", "indexonlyscan", "indexscan"] */
        case 1:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            break;

        /*  2: ["hashjoin", "indexonlyscan", "indexscan", "mergejoin"] */
        case 2:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            break;

        /*  3: ["hashjoin", "indexonlyscan", "indexscan", "mergejoin", "nestloop"] */
        case 3:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /*  4: ["hashjoin", "indexonlyscan", "indexscan", "mergejoin", "seqscan"] */
        case 4:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /*  5: ["hashjoin", "indexonlyscan", "indexscan", "nestloop"] */
        case 5:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_nestloop = true;
            break;

        /*  6: ["hashjoin", "indexonlyscan", "indexscan", "nestloop", "seqscan"] */
        case 6:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /*  7: ["hashjoin", "indexonlyscan", "indexscan", "seqscan"] */
        case 7:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_seqscan = true;
            break;

        /*  8: ["hashjoin", "indexonlyscan", "mergejoin"] */
        case 8:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            break;

        /*  9: ["hashjoin", "indexonlyscan", "mergejoin", "nestloop"] */
        case 9:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /* 10: ["hashjoin", "indexonlyscan", "mergejoin", "nestloop", "seqscan"] */
        case 10:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 11: ["hashjoin", "indexonlyscan", "mergejoin", "seqscan"] */
        case 11:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 12: ["hashjoin", "indexonlyscan", "nestloop"] */
        case 12:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_nestloop = true;
            break;

        /* 13: ["hashjoin", "indexonlyscan", "nestloop", "seqscan"] */
        case 13:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 14: ["hashjoin", "indexonlyscan", "seqscan"] */
        case 14:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            enable_seqscan = true;
            break;

        /* 15: ["hashjoin", "indexscan"] */
        case 15:
            enable_hashjoin = true;
            enable_indexscan = true;
            break;

        /* 16: ["hashjoin", "indexscan", "mergejoin"] */
        case 16:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            break;

        /* 17: ["hashjoin", "indexscan", "mergejoin", "nestloop"] */
        case 17:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /* 18: ["hashjoin", "indexscan", "mergejoin", "nestloop", "seqscan"] */
        case 18:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 19: ["hashjoin", "indexscan", "mergejoin", "seqscan"] */
        case 19:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 20: ["hashjoin", "indexscan", "nestloop"] */
        case 20:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_nestloop = true;
            break;

        /* 21: ["hashjoin", "indexscan", "nestloop", "seqscan"] */
        case 21:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 22: ["hashjoin", "indexscan", "seqscan"] */
        case 22:
            enable_hashjoin = true;
            enable_indexscan = true;
            enable_seqscan = true;
            break;

        /* 23: ["hashjoin", "mergejoin", "nestloop", "seqscan"] */
        case 23:
            enable_hashjoin = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 24: ["hashjoin", "mergejoin", "seqscan"] */
        case 24:
            enable_hashjoin = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 25: ["hashjoin", "nestloop", "seqscan"] */
        case 25:
            enable_hashjoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 26: ["hashjoin", "seqscan"] */
        case 26:
            enable_hashjoin = true;
            enable_seqscan = true;
            break;

        /* 27: ["indexonlyscan", "indexscan", "mergejoin"] */
        case 27:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            break;

        /* 28: ["indexonlyscan", "indexscan", "mergejoin", "nestloop"] */
        case 28:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /* 29: ["indexonlyscan", "indexscan", "mergejoin", "nestloop", "seqscan"] */
        case 29:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 30: ["indexonlyscan", "indexscan", "mergejoin", "seqscan"] */
        case 30:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 31: ["indexonlyscan", "indexscan", "nestloop"] */
        case 31:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_nestloop = true;
            break;

        /* 32: ["indexonlyscan", "indexscan", "nestloop", "seqscan"] */
        case 32:
            enable_indexonlyscan = true;
            enable_indexscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 33: ["indexonlyscan", "mergejoin"] */
        case 33:
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            break;

        /* 34: ["indexonlyscan", "mergejoin", "nestloop"] */
        case 34:
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /* 35: ["indexonlyscan", "mergejoin", "nestloop", "seqscan"] */
        case 35:
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 36: ["indexonlyscan", "mergejoin", "seqscan"] */
        case 36:
            enable_indexonlyscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 37: ["indexonlyscan", "nestloop"] */
        case 37:
            enable_indexonlyscan = true;
            enable_nestloop = true;
            break;

        /* 38: ["indexonlyscan", "nestloop", "seqscan"] */
        case 38:
            enable_indexonlyscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 39: ["indexscan", "mergejoin"] */
        case 39:
            enable_indexscan = true;
            enable_mergejoin = true;
            break;

        /* 40: ["indexscan", "mergejoin", "nestloop"] */
        case 40:
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            break;

        /* 41: ["indexscan", "mergejoin", "nestloop", "seqscan"] */
        case 41:
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 42: ["indexscan", "mergejoin", "seqscan"] */
        case 42:
            enable_indexscan = true;
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 43: ["indexscan", "nestloop"] */
        case 43:
            enable_indexscan = true;
            enable_nestloop = true;
            break;

        /* 44: ["indexscan", "nestloop", "seqscan"] */
        case 44:
            enable_indexscan = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 45: ["mergejoin", "nestloop", "seqscan"] */
        case 45:
            enable_mergejoin = true;
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /* 46: ["mergejoin", "seqscan"] */
        case 46:
            enable_mergejoin = true;
            enable_seqscan = true;
            break;

        /* 47: ["nestloop", "seqscan"] */
        case 47:
            enable_nestloop = true;
            enable_seqscan = true;
            break;

        /*  48: ["hashjoin", "indexonlyscan"] */
        case 48:
            enable_hashjoin = true;
            enable_indexonlyscan = true;
            break;



        default:
            /* The Python code raises an exception; here we do an elog(ERROR). */
            elog(ERROR,
                 "RegBlocker only supports 49 arms (0..48). Got arm_idx=%d",
                 arm);
            break;
    }
}

// Get a query plan for a particular arm.
static PlannedStmt* plan_arm(int arm, Query* parse,
  int cursorOptions, ParamListInfo boundParams) {

PlannedStmt* plan = NULL;
Query* query_copy = copyObject(parse); // create a copy of the query plan

if (arm == -1) {
// Use whatever the user has set as the current configuration.
plan = standard_planner(query_copy, cursorOptions, boundParams);
return plan;
}

// Preserving the user's options, set the config to match the arm index
// and invoke the PG planner.
save_arm_options({
set_arm_options(arm);
plan = standard_planner(query_copy, cursorOptions, boundParams);
});

return plan;
}

// A struct to represent a query plan before we transform it into JSON.
typedef struct BaoPlanNode {
  // An integer representation of the PG NodeTag.
  unsigned int node_type;

  // The optimizer cost for this node (total cost).
  double optimizer_cost;

  // The cardinality estimate (plan rows) for this node.
  double cardinality_estimate;

  // If this is a scan or index lookup, the name of the underlying relation.
  char* relation_name;

  // Left child.
  struct BaoPlanNode* left;


  // Right child.
  struct BaoPlanNode* right;
} BaoPlanNode;

// Transform the operator types we care about from their PG tag to a
// string. Call other operators "Other".
static const char* node_type_to_string(NodeTag tag) {
  switch (tag) {
  case T_SeqScan:
    return "Seq Scan";
  case T_IndexScan:
    return "Index Scan";
  case T_IndexOnlyScan:
    return "Index Only Scan";
  case T_BitmapIndexScan:
    return "Bitmap Index Scan";
  case T_NestLoop:
    return "Nested Loop";
  case T_MergeJoin:
    return "Merge Join";
  case T_HashJoin:
    return "Hash Join";
  default:
    return "Other";
  }
}

// Allocate an empty BaoPlanNode.
static BaoPlanNode* new_bao_plan() {
  return (BaoPlanNode*) malloc(sizeof(BaoPlanNode));
}

// Free (recursively) an entire BaoPlanNode. Frees children as well.
static void free_bao_plan_node(BaoPlanNode* node) {
  if (node->left) free_bao_plan_node(node->left);
  if (node->right) free_bao_plan_node(node->right);
  free(node);
}

// Emit a JSON representation of the given BaoPlanNode to the stream given.
// Recursive function, the entry point is `plan_to_json`.
static void emit_json(BaoPlanNode* node, FILE* stream) {
  fprintf(stream, "{\"Node Type\": \"%s\",", node_type_to_string(node->node_type));
  fprintf(stream, "\"Node Type ID\": \"%d\",", node->node_type);
  if (node->relation_name)
    // TODO need to escape the relation name for JSON...
    fprintf(stream, "\"Relation Name\": \"%s\",", node->relation_name);
  fprintf(stream, "\"Total Cost\": %f,", node->optimizer_cost);
  fprintf(stream, "\"Plan Rows\": %f", node->cardinality_estimate);
  if (!node->left && !node->right) {
    fprintf(stream, "}");
    return;
  }

  fprintf(stream, ", \"Plans\": [");
  if (node->left) emit_json(node->left, stream);
  if (node->right) {
    fprintf(stream, ", ");
    emit_json(node->right, stream);
  }
  fprintf(stream, "]}");
}

// Transform a PostgreSQL PlannedStmt into a BaoPlanNode tree.
static BaoPlanNode* transform_plan(PlannedStmt* stmt, Plan* node) {
  BaoPlanNode* result = new_bao_plan();

  result->node_type = node->type;
  result->optimizer_cost = node->total_cost;
  result->cardinality_estimate = node->plan_rows;
  result->relation_name = get_relation_name(stmt, node);

  result->left = NULL;
  result->right = NULL;
  if (node->lefttree) result->left = transform_plan(stmt, node->lefttree);
  if (node->righttree) result->right = transform_plan(stmt, node->righttree);

  return result;
}

// Given a PostgreSQL PlannedStmt, produce the JSON representation we need to
// send to the Bao server.
static char* plan_to_json(PlannedStmt* plan) {
  char* buf;
  size_t json_size;
  FILE* stream;
  BaoPlanNode* transformed_plan;

  transformed_plan = transform_plan(plan, plan->planTree);
  
  stream = open_memstream(&buf, &json_size);
  fprintf(stream, "{\"Plan\": ");
  emit_json(transformed_plan, stream);
  fprintf(stream, "}\n");
  fclose(stream);

  free_bao_plan_node(transformed_plan);
  
  return buf;
}

// Primary planning function. Invokes the PG planner for each arm, sends the
// results to the Bao server, gets the response, and returns the corrosponding
// query plan (as a BaoPlan).
BaoPlan* plan_query(Query *parse, int cursorOptions, ParamListInfo boundParams) {
  BaoPlan* plan;
  PlannedStmt* plan_for_arm[BAO_MAX_ARMS];
  char* json_for_arm[BAO_MAX_ARMS];
  Query* query_copy;
  int conn_fd;

  // Prepare the plan object to store a BaoQueryInfo instance.
  plan = (BaoPlan*) malloc(sizeof(BaoPlan));
  plan->query_info = (BaoQueryInfo*) malloc(sizeof(BaoQueryInfo));
  plan->selection = 0;
  
  // Connect this buffer state with the query.
  plan->query_info->buffer_json = buffer_state();

  if (!enable_bao_selection) {
    // If Bao is not picking query plans, we use arm -1 to get the
    // default PostgreSQL plan. Note that we do *not* use arm 0, as
    // this would ignore the user's settings for things like
    // enable_nestloop.
    plan->plan = plan_arm(-1, parse, cursorOptions, boundParams);
    plan->query_info->plan_json = plan_to_json(plan->plan);
    return plan;
  }
  
  conn_fd = connect_to_bao(bao_host, bao_port);
  if (conn_fd == -1) {
    elog(WARNING, "Unable to connect to Bao server.");
    return NULL;
  }

  memset(plan_for_arm, 0, BAO_MAX_ARMS*sizeof(PlannedStmt*));

  write_all_to_socket(conn_fd, START_QUERY_MESSAGE);
  for (int i = 0; i < bao_num_arms; i++) {
    // Plan the query for this arm.
    query_copy = copyObject(parse);
    plan_for_arm[i] = plan_arm(i, query_copy, cursorOptions, boundParams);

    // Transform it into JSON, transmit it to the Bao server.
    json_for_arm[i] = plan_to_json(plan_for_arm[i]);
    write_all_to_socket(conn_fd, json_for_arm[i]);
  }
  
  write_all_to_socket(conn_fd, plan->query_info->buffer_json);
  write_all_to_socket(conn_fd, TERMINAL_MESSAGE);
  shutdown(conn_fd, SHUT_WR);

  // Read the response.
  if (read(conn_fd, &plan->selection, sizeof(unsigned int)) != sizeof(unsigned int)) {
    shutdown(conn_fd, SHUT_RDWR);
    elog(WARNING, "Bao could not read the response from the server.");
    plan->selection = 0;
  }
  shutdown(conn_fd, SHUT_RDWR);

  if (plan->selection >= BAO_MAX_ARMS) {
    elog(ERROR, "Bao server returned arm index %d, which is outside the range.",
         plan->selection);
    plan->selection = 0;
  }
  
  // Keep the plan the Bao server selected, and associate the JSON representation
  // of the plan with the BaoPlan. Free everything else.
  plan->plan = plan_for_arm[plan->selection];
  for (int i = 0; i < bao_num_arms; i++) {
    if (i == plan->selection) {
      plan->query_info->plan_json = json_for_arm[i];
    } else {
      free(json_for_arm[i]);
    }
  }
    
  return plan;
}

// Given an arm index, produce the SQL statements that would cause PostgreSQL to
// select the same query plan as Bao would.
char* arm_to_hint(int arm) {
  char* buf;
  size_t size;
  FILE* stream;
  
  stream = open_memstream(&buf, &size);

  save_arm_options({
      set_arm_options(arm);
      if (!enable_nestloop) fprintf(stream, "SET enable_nestloop TO off; ");
      if (!enable_hashjoin) fprintf(stream, "SET enable_hashjoin TO off; ");
      if (!enable_mergejoin) fprintf(stream, "SET enable_mergejoin TO off; ");
      if (!enable_seqscan) fprintf(stream, "SET enable_seqscan TO off; ");
      if (!enable_indexscan) fprintf(stream, "SET enable_indexscan TO off; ");
      if (!enable_indexonlyscan) fprintf(stream, "SET enable_indexonlyscan TO off; ");
    });

  fclose(stream);
  
  if (size == 0) return NULL;
  return buf;
}

#endif
