/*
 * apus encoding.h — DeepSeek-V4 message format, C11.
 *
 * Faithful C port of reference-0731/encoding/encoding_dsv4.py (encode path;
 * the 0731 official release): roles (system, user, assistant, tool-merging,
 * latest_reminder, developer), thinking vs chat mode, DSML tool-call
 * rendering, tool-result merging and
 * call-order sorting, reasoning_effort low/high/max prefixes, BOS handling,
 * generation-prompt construction. JSON serialization matches Python's
 * json.dumps(ensure_ascii=False) byte-for-byte (via json.h).
 *
 * reasoning_effort (0731 semantics): "low" (default; NULL maps to it) adds
 * no prefix; "high" prepends the OLD preview "max" prompt ("Absolute
 * maximum..."); "max" prepends the NEW 0731 prompt ("Beyond maximum...").
 * Thinking mode only, at message index 0.
 *
 * The message input is a json.h tree: a J_ARR of J_OBJ messages using the
 * same field names as the Python reference (OpenAI-style messages).
 *
 * Usage: #define APUS_ENCODING_IMPLEMENTATION in exactly one TU.
 */
#ifndef APUS_ENCODING_H
#define APUS_ENCODING_H

#include <stddef.h>
#include <stdint.h>

#include "json.h"
#include "tok.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *thinking_mode;    /* "chat" | "thinking" (required) */
    int         drop_thinking;    /* default 1 */
    int         add_bos;          /* default 1 */
    const char *reasoning_effort; /* NULL | "low" | "high" | "max"
                                     (NULL == "low", no prefix) */
} DsEncOpts;

#define DS_ENC_OPTS_DEFAULT { "thinking", 1, 1, NULL }

/* Encode a message list to the prompt string (malloc'd; NULL on error,
 * see ds_last_error()). context may be NULL. */
char *ds_encode_messages(const JVal *messages, const DsEncOpts *opts);
char *ds_encode_messages_ctx(const JVal *messages, const JVal *context,
                             const DsEncOpts *opts);

/* Encode straight to token ids (combines string assembly with tok.h).
 * Special tokens in the assembled prompt are recognized. */
uint32_t *ds_encode_ids(const Tok *t, const JVal *messages,
                        const DsEncOpts *opts, size_t *n_out);

const char *ds_last_error(void);

/* Special token strings (exact UTF-8) */
extern const char *DS_BOS;
extern const char *DS_EOS;
extern const char *DS_THINK_START;
extern const char *DS_THINK_END;
extern const char *DS_DSML;
extern const char *DS_USER_SP;
extern const char *DS_ASSISTANT_SP;
extern const char *DS_LATEST_REMINDER_SP;

#ifdef __cplusplus
}
#endif

#endif /* APUS_ENCODING_H */

/* ================================================================== */
#if defined(APUS_ENCODING_IMPLEMENTATION) && !defined(APUS_ENCODING_IMPL_DONE)
#define APUS_ENCODING_IMPL_DONE

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>

const char *DS_BOS = "<｜begin▁of▁sentence｜>";
const char *DS_EOS = "<｜end▁of▁sentence｜>";
const char *DS_THINK_START = "<think>";
const char *DS_THINK_END = "</think>";
const char *DS_DSML = "｜DSML｜";
const char *DS_USER_SP = "<｜User｜>";
const char *DS_ASSISTANT_SP = "<｜Assistant｜>";
const char *DS_LATEST_REMINDER_SP = "<｜latest_reminder｜>";

static const char *DS_TASK_SP_ACTION = "<｜action｜>";
static const char *DS_TASK_SP_QUERY = "<｜query｜>";
static const char *DS_TASK_SP_AUTHORITY = "<｜authority｜>";
static const char *DS_TASK_SP_DOMAIN = "<｜domain｜>";
static const char *DS_TASK_SP_TITLE = "<｜title｜>";
static const char *DS_TASK_SP_READ_URL = "<｜read_url｜>";

/* Reasoning effort prompts (0731): prepended at index 0 in thinking mode.
 * "low" (the default) adds nothing. "high" is the OLD preview "max" text;
 * "max" is the NEW 0731 text. */
static const char *REASONING_EFFORT_HIGH =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

static const char *REASONING_EFFORT_MAX =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

static const char *RESPONSE_FORMAT_TEMPLATE =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

static const char *TOOLS_TEMPLATE_PART1 =
    "## Tools\n"
    "\n"
    "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n"
    "\n"
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "...\n"
    "</｜DSML｜invoke>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
    "...\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n"
    "\n"
    "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n"
    "\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n"
    "\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n"
    "\n"
    "### Available Tool Schemas\n"
    "\n";

static const char *TOOLS_TEMPLATE_PART2 =
    "\n"
    "\n"
    "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

/* ---------------- error reporting ---------------- */

static char ds_err[256];

const char *ds_last_error(void) { return ds_err; }

static int ds_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(ds_err, sizeof ds_err, fmt, ap);
    va_end(ap);
    return -1;
}

/* ---------------- helpers ---------------- */

static const char *jstr(const JVal *obj, const char *key) {
    JVal *v = json_obj_get((JVal *)obj, key);
    return json_type(v) == J_STR ? json_str(v) : NULL;
}

static const char *jrole(const JVal *msg) {
    const char *r = jstr(msg, "role");
    return r ? r : "";
}

static int role_is(const JVal *msg, const char *role) {
    return strcmp(jrole(msg), role) == 0;
}

static int has_task(const JVal *msg) {
    JVal *v = json_obj_get((JVal *)msg, "task");
    return v && json_type(v) != J_NULL;
}

static int find_last_user_index(const JVal *msgs) {
    size_t n = json_arr_len(msgs);
    for (size_t i = n; i-- > 0;) {
        const JVal *m = json_arr_get(msgs, i);
        if (role_is(m, "user") || role_is(m, "developer")) return (int)i;
    }
    return -1;
}

/* Python's to_json: json.dumps(value, ensure_ascii=False) */
static char *to_json(const JVal *v) { return json_dumps(v); }

/* ---------------- DSML argument encoding ---------------- */

static void sb_puts_esc_none(SBuf *b, const char *s) { sb_puts(b, s ? s : ""); }

/* Python: f"[Unsupported {block_type}]" (None -> "None") */
static void sb_unsupported(SBuf *b, const char *type) {
    sb_puts(b, "[Unsupported ");
    sb_puts(b, type ? type : "None");
    sb_puts(b, "]");
}

/* encode_arguments_to_dsml: arguments is a JSON string. */
static char *encode_arguments_to_dsml(const char *arguments) {
    char perr[128];
    JVal *args = json_parse(arguments ? arguments : "", arguments ? strlen(arguments) : 0,
                            perr, sizeof perr);
    if (!args || json_type(args) != J_OBJ) {
        json_free(args);
        args = json_new_obj();
        json_obj_set(args, "arguments", json_new_str(arguments ? arguments : ""));
    }
    SBuf b;
    sb_init(&b);
    size_t n = json_obj_len(args);
    for (size_t i = 0; i < n; i++) {
        const char *k = json_obj_key(args, i);
        JVal *v = json_obj_val(args, i);
        int is_str = json_type(v) == J_STR;
        if (i) sb_putc(&b, '\n');
        sb_puts(&b, "<｜DSML｜parameter name=\"");
        sb_puts_esc_none(&b, k);
        sb_puts(&b, "\" string=\"");
        sb_puts(&b, is_str ? "true" : "false");
        sb_puts(&b, "\">");
        if (is_str) {
            sb_puts(&b, json_str(v));
        } else {
            char *j = to_json(v);
            sb_puts(&b, j);
            free(j);
        }
        sb_puts(&b, "</｜DSML｜parameter>");
    }
    json_free(args);
    return sb_steal(&b);
}

/* ---------------- tools rendering ---------------- */

static char *render_tools(const JVal *tools) {
    /* tools_from_openai_format: [tool["function"] for tool in tools] */
    SBuf schemas;
    sb_init(&schemas);
    size_t n = json_arr_len(tools);
    for (size_t i = 0; i < n; i++) {
        JVal *t = json_arr_get(tools, i);
        JVal *fn = t ? json_obj_get(t, "function") : NULL;
        char *j = to_json(fn);
        if (i) sb_putc(&schemas, '\n');
        sb_puts(&schemas, j);
        free(j);
    }
    SBuf b;
    sb_init(&b);
    sb_puts(&b, TOOLS_TEMPLATE_PART1);
    char *sch = sb_steal(&schemas);
    sb_puts(&b, sch);
    free(sch);
    sb_puts(&b, TOOLS_TEMPLATE_PART2);
    return sb_steal(&b);
}

/* ---------------- merge_tool_messages ---------------- */

static JVal *merge_tool_messages(const JVal *messages) {
    JVal *merged = json_new_arr();
    size_t n = json_arr_len(messages);
    for (size_t i = 0; i < n; i++) {
        const JVal *msg = json_arr_get(messages, i);
        if (!msg || json_type(msg) != J_OBJ) {
            json_arr_push(merged, json_clone(msg));
            continue;
        }
        if (role_is(msg, "tool")) {
            JVal *block = json_new_obj();
            json_obj_set(block, "type", json_new_str("tool_result"));
            const char *tcid = jstr(msg, "tool_call_id");
            json_obj_set(block, "tool_use_id", json_new_str(tcid ? tcid : ""));
            JVal *content = json_obj_get((JVal *)msg, "content");
            json_obj_set(block, "content",
                         content ? json_clone(content) : json_new_str(""));
            size_t m = json_arr_len(merged);
            JVal *last = m ? json_arr_get(merged, m - 1) : NULL;
            if (last && role_is(last, "user") && json_obj_get(last, "content_blocks")) {
                json_arr_push(json_obj_get(last, "content_blocks"), block);
            } else {
                JVal *u = json_new_obj();
                json_obj_set(u, "role", json_new_str("user"));
                JVal *cbs = json_new_arr();
                json_arr_push(cbs, block);
                json_obj_set(u, "content_blocks", cbs);
                json_arr_push(merged, u);
            }
        } else if (role_is(msg, "user")) {
            JVal *text_block = json_new_obj();
            json_obj_set(text_block, "type", json_new_str("text"));
            JVal *content = json_obj_get((JVal *)msg, "content");
            json_obj_set(text_block, "text",
                         content && json_type(content) == J_STR
                             ? json_clone(content)
                             : json_new_str(""));
            size_t m = json_arr_len(merged);
            JVal *last = m ? json_arr_get(merged, m - 1) : NULL;
            if (last && role_is(last, "user") && json_obj_get(last, "content_blocks") &&
                !has_task(last)) {
                json_arr_push(json_obj_get(last, "content_blocks"), text_block);
            } else {
                JVal *u = json_new_obj();
                json_obj_set(u, "role", json_new_str("user"));
                json_obj_set(u, "content",
                             content && json_type(content) == J_STR
                                 ? json_clone(content)
                                 : json_new_str(""));
                JVal *cbs = json_new_arr();
                json_arr_push(cbs, text_block);
                json_obj_set(u, "content_blocks", cbs);
                const char *extra[] = { "task", "wo_eos", "mask" };
                for (int k = 0; k < 3; k++) {
                    JVal *v = json_obj_get((JVal *)msg, extra[k]);
                    if (v) json_obj_set(u, extra[k], json_clone(v));
                }
                json_arr_push(merged, u);
            }
        } else {
            json_arr_push(merged, json_clone(msg));
        }
    }
    return merged;
}

/* ---------------- sort_tool_results_by_call_order ---------------- */

static void sort_tool_results_by_call_order(JVal *messages) {
    /* id -> order map of the last assistant message with tool_calls */
    char **ids = NULL;
    int *order = NULL;
    size_t n_ids = 0, cap = 0;

    size_t n = json_arr_len(messages);
    for (size_t i = 0; i < n; i++) {
        JVal *msg = json_arr_get(messages, i);
        if (!msg || json_type(msg) != J_OBJ) continue;
        if (role_is(msg, "assistant")) {
            JVal *tcs = json_obj_get(msg, "tool_calls");
            if (tcs && json_truthy(tcs)) {
                n_ids = 0;
                size_t m = json_arr_len(tcs);
                for (size_t j = 0; j < m; j++) {
                    JVal *tc = json_arr_get(tcs, j);
                    if (!tc || json_type(tc) != J_OBJ) continue;
                    const char *id = jstr(tc, "id");
                    if (!id || !id[0]) {
                        JVal *fn = json_obj_get(tc, "function");
                        id = fn && json_type(fn) == J_OBJ ? jstr(fn, "id") : NULL;
                    }
                    if (id && id[0]) {
                        /* Python dict: duplicate ids keep the LAST order */
                        size_t slot = n_ids;
                        for (size_t k = 0; k < n_ids; k++)
                            if (strcmp(ids[k], id) == 0) { slot = k; break; }
                        if (slot == n_ids) {
                            if (n_ids == cap) {
                                cap = cap ? cap * 2 : 8;
                                ids = (char **)realloc(ids, cap * sizeof(char *));
                                order = (int *)realloc(order, cap * sizeof(int));
                            }
                            ids[n_ids] = (char *)id;
                            n_ids++;
                        }
                        order[slot] = (int)j;
                    }
                }
            }
        } else if (role_is(msg, "user")) {
            JVal *cbs = json_obj_get(msg, "content_blocks");
            if (!cbs || json_type(cbs) != J_ARR) continue;
            size_t m = json_arr_len(cbs), n_tool = 0;
            for (size_t j = 0; j < m; j++) {
                const char *tp = jstr(json_arr_get(cbs, j), "type");
                if (tp && strcmp(tp, "tool_result") == 0) n_tool++;
            }
            if (n_tool > 1 && n_ids > 0) {
                /* stable sort of the tool_result blocks by call order,
                   written back into their original slots */
                size_t *pos = (size_t *)malloc(n_tool * sizeof(size_t));
                size_t *keys = (size_t *)malloc(n_tool * sizeof(size_t));
                JVal **blocks = (JVal **)malloc(n_tool * sizeof(JVal *));
                size_t ti = 0;
                for (size_t j = 0; j < m; j++) {
                    JVal *b = json_arr_get(cbs, j);
                    const char *tp = jstr(b, "type");
                    if (tp && strcmp(tp, "tool_result") == 0) {
                        const char *tuid = jstr(b, "tool_use_id");
                        size_t key = 0;
                        if (tuid)
                            for (size_t k = 0; k < n_ids; k++)
                                if (strcmp(ids[k], tuid) == 0) { key = (size_t)order[k]; break; }
                        pos[ti] = j;
                        keys[ti] = key;
                        blocks[ti] = json_clone(b);
                        ti++;
                    }
                }
                for (size_t a = 1; a < n_tool; a++) {   /* insertion sort: stable */
                    size_t kv = keys[a];
                    JVal *bv = blocks[a];
                    size_t b = a;
                    while (b > 0 && keys[b - 1] > kv) {
                        keys[b] = keys[b - 1];
                        blocks[b] = blocks[b - 1];
                        b--;
                    }
                    keys[b] = kv;
                    blocks[b] = bv;
                }
                for (size_t a = 0; a < n_tool; a++)
                    json_arr_replace(cbs, pos[a], blocks[a]);
                free(pos);
                free(keys);
                free(blocks);
            }
        }
    }
    free(ids);
    free(order);
}

/* ---------------- _drop_thinking_messages ---------------- */

static int in_keep_roles(const char *role) {
    return strcmp(role, "user") == 0 || strcmp(role, "system") == 0 ||
           strcmp(role, "tool") == 0 || strcmp(role, "latest_reminder") == 0 ||
           strcmp(role, "direct_search_results") == 0;
}

static JVal *drop_thinking_messages(const JVal *messages) {
    int last_user_idx = find_last_user_index(messages);
    JVal *result = json_new_arr();
    size_t n = json_arr_len(messages);
    for (size_t i = 0; i < n; i++) {
        const JVal *msg = json_arr_get(messages, i);
        const char *role = jrole(msg);
        if (in_keep_roles(role) || (int)i >= last_user_idx) {
            json_arr_push(result, json_clone(msg));
        } else if (strcmp(role, "assistant") == 0) {
            JVal *copy = json_clone(msg);
            json_obj_del(copy, "reasoning_content");
            json_arr_push(result, copy);
        }
        /* developer and other roles before last_user_idx are dropped */
    }
    return result;
}

/* ---------------- render_message ---------------- */

static const char *task_sp_token(const char *task) {
    if (strcmp(task, "action") == 0) return DS_TASK_SP_ACTION;
    if (strcmp(task, "query") == 0) return DS_TASK_SP_QUERY;
    if (strcmp(task, "authority") == 0) return DS_TASK_SP_AUTHORITY;
    if (strcmp(task, "domain") == 0) return DS_TASK_SP_DOMAIN;
    if (strcmp(task, "title") == 0) return DS_TASK_SP_TITLE;
    if (strcmp(task, "read_url") == 0) return DS_TASK_SP_READ_URL;
    return NULL;
}

static int render_message(SBuf *out, size_t index, const JVal *messages,
                          const char *thinking_mode, int drop_thinking,
                          const char *reasoning_effort) {
    size_t n = json_arr_len(messages);
    const JVal *msg = json_arr_get(messages, index);
    int last_user_idx = find_last_user_index(messages);
    int thinking = strcmp(thinking_mode, "thinking") == 0;

    const char *role = jrole(msg);
    const char *content = jstr(msg, "content");
    JVal *tools = json_obj_get((JVal *)msg, "tools");
    JVal *response_format = json_obj_get((JVal *)msg, "response_format");
    JVal *tool_calls = json_obj_get((JVal *)msg, "tool_calls");
    const char *reasoning_content = jstr(msg, "reasoning_content");
    int wo_eos = json_truthy(json_obj_get((JVal *)msg, "wo_eos"));

    /* Reasoning effort prefix (only at index 0 in thinking mode; "low"/NULL
     * adds nothing) — reference-0731 encoding_dsv4.py REASONING_EFFORT_PROMPTS */
    if (index == 0 && thinking && reasoning_effort) {
        if (strcmp(reasoning_effort, "high") == 0)
            sb_puts(out, REASONING_EFFORT_HIGH);
        else if (strcmp(reasoning_effort, "max") == 0)
            sb_puts(out, REASONING_EFFORT_MAX);
    }

    if (strcmp(role, "system") == 0) {
        sb_puts_esc_none(out, content);
        if (tools && json_truthy(tools)) {
            char *rt = render_tools(tools);
            sb_puts(out, "\n\n");
            sb_puts(out, rt);
            free(rt);
        }
        if (response_format && json_truthy(response_format)) {
            char *j = to_json(response_format);
            sb_puts(out, "\n\n");
            sb_puts(out, RESPONSE_FORMAT_TEMPLATE);
            sb_puts(out, j);
            free(j);
        }
    } else if (strcmp(role, "developer") == 0) {
        if (!content || !content[0])
            return ds_fail("invalid developer message (empty content)");
        sb_puts(out, DS_USER_SP);
        sb_puts(out, content);
        if (tools && json_truthy(tools)) {
            char *rt = render_tools(tools);
            sb_puts(out, "\n\n");
            sb_puts(out, rt);
            free(rt);
        }
        if (response_format && json_truthy(response_format)) {
            char *j = to_json(response_format);
            sb_puts(out, "\n\n");
            sb_puts(out, RESPONSE_FORMAT_TEMPLATE);
            sb_puts(out, j);
            free(j);
        }
    } else if (strcmp(role, "user") == 0) {
        sb_puts(out, DS_USER_SP);
        JVal *cbs = json_obj_get((JVal *)msg, "content_blocks");
        if (cbs && json_truthy(cbs)) {
            size_t m = json_arr_len(cbs);
            for (size_t i = 0; i < m; i++) {
                JVal *block = json_arr_get(cbs, i);
                const char *bt = (block && json_type(block) == J_OBJ) ? jstr(block, "type") : NULL;
                if (i) sb_puts(out, "\n\n");
                if (bt && strcmp(bt, "text") == 0) {
                    sb_puts_esc_none(out, jstr(block, "text"));
                } else if (bt && strcmp(bt, "tool_result") == 0) {
                    JVal *tc = json_obj_get(block, "content");
                    sb_puts(out, "<tool_result>");
                    if (tc && json_type(tc) == J_ARR) {
                        size_t qn = json_arr_len(tc);
                        for (size_t q = 0; q < qn; q++) {
                            JVal *sub = json_arr_get(tc, q);
                            const char *st = (sub && json_type(sub) == J_OBJ) ? jstr(sub, "type") : NULL;
                            if (q) sb_puts(out, "\n\n");
                            if (st && strcmp(st, "text") == 0)
                                sb_puts_esc_none(out, jstr(sub, "text"));
                            else
                                sb_unsupported(out, st);
                        }
                    } else if (tc && json_type(tc) == J_STR) {
                        sb_puts(out, json_str(tc));
                    } else if (tc && json_type(tc) == J_NULL) {
                        /* Python: template.format(content=None) -> "None" */
                        sb_puts(out, "None");
                    }
                    sb_puts(out, "</tool_result>");
                } else {
                    sb_unsupported(out, bt);
                }
            }
        } else {
            sb_puts_esc_none(out, content);
        }
    } else if (strcmp(role, "latest_reminder") == 0) {
        sb_puts(out, DS_LATEST_REMINDER_SP);
        /* Python: template.format(content=content) -> "None" if null */
        sb_puts(out, content ? content : "None");
    } else if (strcmp(role, "tool") == 0) {
        return ds_fail("deepseek_v4 merges tool messages into user; "
                       "preprocess with merge_tool_messages()");
    } else if (strcmp(role, "assistant") == 0) {
        SBuf thinking_part;
        sb_init(&thinking_part);
        SBuf tc;
        sb_init(&tc);
        if (tool_calls && json_truthy(tool_calls)) {
            size_t m = json_arr_len(tool_calls);
            sb_puts(&tc, "\n\n<｜DSML｜tool_calls>\n");
            for (size_t i = 0; i < m; i++) {
                JVal *call = json_arr_get(tool_calls, i);
                JVal *fn = call && json_type(call) == J_OBJ ? json_obj_get(call, "function") : NULL;
                const char *name = fn && json_type(fn) == J_OBJ ? jstr(fn, "name") : NULL;
                const char *args = fn && json_type(fn) == J_OBJ ? jstr(fn, "arguments") : NULL;
                char *dsml = encode_arguments_to_dsml(args);
                if (i) sb_putc(&tc, '\n');
                sb_puts(&tc, "<｜DSML｜invoke name=\"");
                sb_puts_esc_none(&tc, name);
                sb_puts(&tc, "\">\n");
                sb_puts(&tc, dsml);
                sb_puts(&tc, "\n</｜DSML｜invoke>");
                free(dsml);
            }
            sb_puts(&tc, "\n</｜DSML｜tool_calls>");
        }
        const char *summary = content ? content : "";
        const char *rc = reasoning_content ? reasoning_content : "";
        int prev_has_task = index > 0 && has_task(json_arr_get(messages, index - 1));
        if (thinking && !prev_has_task) {
            if (!drop_thinking || (int)index > last_user_idx) {
                sb_puts(&thinking_part, rc);
                sb_puts(&thinking_part, DS_THINK_END);
            }
        }
        char *tp = sb_steal(&thinking_part);
        sb_puts(out, tp);
        free(tp);
        sb_puts(out, summary);
        char *tcs = sb_steal(&tc);
        sb_puts(out, tcs);
        free(tcs);
        if (!wo_eos) sb_puts(out, DS_EOS);
    } else {
        return ds_fail("unknown role: %s", role);
    }

    /* transition tokens based on what follows */
    if (index + 1 < n) {
        const char *next_role = jrole(json_arr_get(messages, index + 1));
        if (strcmp(next_role, "assistant") != 0 &&
            strcmp(next_role, "latest_reminder") != 0)
            return 0;
    }

    const char *task = jstr(msg, "task");
    JVal *task_v = json_obj_get((JVal *)msg, "task");
    if (task_v && json_type(task_v) != J_NULL) {
        if (!task || !task_sp_token(task))
            return ds_fail("invalid task: %s", task ? task : "(non-string)");
        if (strcmp(task, "action") != 0) {
            sb_puts(out, task_sp_token(task));
        } else {
            sb_puts(out, DS_ASSISTANT_SP);
            sb_puts(out, thinking ? DS_THINK_START : DS_THINK_END);
            sb_puts(out, DS_TASK_SP_ACTION);
        }
    } else if (strcmp(role, "user") == 0 || strcmp(role, "developer") == 0) {
        sb_puts(out, DS_ASSISTANT_SP);
        if (!drop_thinking && thinking)
            sb_puts(out, DS_THINK_START);
        else if (drop_thinking && thinking && (int)index >= last_user_idx)
            sb_puts(out, DS_THINK_START);
        else
            sb_puts(out, DS_THINK_END);
    }
    return 0;
}

/* ---------------- encode_messages ---------------- */

static int any_tools(const JVal *msgs) {
    size_t n = json_arr_len(msgs);
    for (size_t i = 0; i < n; i++) {
        JVal *v = json_obj_get(json_arr_get(msgs, i), "tools");
        if (v && json_truthy(v)) return 1;
    }
    return 0;
}

static JVal *arr_concat_clone(const JVal *a, const JVal *b) {
    JVal *r = json_new_arr();
    size_t na = a ? json_arr_len(a) : 0;
    size_t nb = b ? json_arr_len(b) : 0;
    for (size_t i = 0; i < na; i++) json_arr_push(r, json_clone(json_arr_get(a, i)));
    for (size_t i = 0; i < nb; i++) json_arr_push(r, json_clone(json_arr_get(b, i)));
    return r;
}

char *ds_encode_messages_ctx(const JVal *messages, const JVal *context,
                             const DsEncOpts *opts) {
    ds_err[0] = '\0';
    if (!messages || json_type(messages) != J_ARR) {
        ds_fail("messages must be a JSON array");
        return NULL;
    }
    const char *thinking_mode = opts && opts->thinking_mode ? opts->thinking_mode : "thinking";
    if (strcmp(thinking_mode, "chat") != 0 && strcmp(thinking_mode, "thinking") != 0) {
        ds_fail("invalid thinking_mode `%s`", thinking_mode);
        return NULL;
    }
    const char *effort = opts ? opts->reasoning_effort : NULL;
    if (effort && strcmp(effort, "low") != 0 && strcmp(effort, "high") != 0
        && strcmp(effort, "max") != 0) {
        ds_fail("invalid reasoning effort: %s", effort);
        return NULL;
    }
    int drop_thinking = opts ? opts->drop_thinking : 1;
    int add_bos = opts ? opts->add_bos : 1;
    int thinking = strcmp(thinking_mode, "thinking") == 0;

    JVal *merged = merge_tool_messages(messages);

    /* sort with the context prefix, then drop the prefix */
    size_t ctx_in_n = context && json_type(context) == J_ARR ? json_arr_len(context) : 0;
    JVal *sorted_full = arr_concat_clone(context, merged);
    json_free(merged);
    sort_tool_results_by_call_order(sorted_full);
    JVal *msgs = json_new_arr();
    for (size_t i = ctx_in_n; i < json_arr_len(sorted_full); i++)
        json_arr_push(msgs, json_clone(json_arr_get(sorted_full, i)));
    json_free(sorted_full);

    JVal *ctx = NULL;
    if (ctx_in_n) {
        ctx = merge_tool_messages(context);
        sort_tool_results_by_call_order(ctx);
    }

    JVal *full = arr_concat_clone(ctx, msgs);
    size_t ctx_n = ctx ? json_arr_len(ctx) : 0;

    SBuf out;
    sb_init(&out);
    if (add_bos && ctx_n == 0) sb_puts(&out, DS_BOS);

    int eff_drop = drop_thinking && !any_tools(full);

    int rc = 0;
    if (thinking && eff_drop) {
        JVal *full_d = drop_thinking_messages(full);
        JVal *ctx_d = ctx ? drop_thinking_messages(ctx) : json_new_arr();
        size_t num_to_render = json_arr_len(full_d) - json_arr_len(ctx_d);
        size_t context_len = json_arr_len(full_d) - num_to_render;
        for (size_t idx = 0; idx < num_to_render && !rc; idx++)
            rc = render_message(&out, idx + context_len, full_d,
                                thinking_mode, eff_drop, effort) < 0;
        json_free(full_d);
        json_free(ctx_d);
    } else {
        size_t num_to_render = json_arr_len(msgs);
        for (size_t idx = 0; idx < num_to_render && !rc; idx++)
            rc = render_message(&out, idx + ctx_n, full,
                                thinking_mode, eff_drop, effort) < 0;
    }

    json_free(msgs);
    json_free(ctx);
    json_free(full);
    if (rc) {
        sb_free(&out);
        return NULL;
    }
    return sb_steal(&out);
}

char *ds_encode_messages(const JVal *messages, const DsEncOpts *opts) {
    return ds_encode_messages_ctx(messages, NULL, opts);
}

uint32_t *ds_encode_ids(const Tok *t, const JVal *messages,
                        const DsEncOpts *opts, size_t *n_out) {
    char *prompt = ds_encode_messages(messages, opts);
    if (!prompt) return NULL;
    uint32_t *ids = tok_encode_str(t, prompt, 1, n_out);
    free(prompt);
    return ids;
}

#endif /* APUS_ENCODING_IMPL_DONE */
