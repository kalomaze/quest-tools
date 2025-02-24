import requests
import json
import math

def get_model_name():
    url = "http://localhost:8000/v1/models"
    try:
        response = requests.get(url)
        models = response.json()
        if models and 'data' in models:
            print(f"Available models: {models['data']}")
            return models['data'][0]['id']  # Return first available model
    except Exception as e:
        print(f"Error getting models: {e}")
        return "Quest-AI/rm-proto-7b-s104-v1"  # Fallback default

def query_vllm(prompt):
    url = "http://localhost:8000/v1/completions"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    model_name = get_model_name()
    print(f"Using model: {model_name}")
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 1.0,
        "logprobs": 20
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()

def logprobs_to_probs(logprobs_dict):
    # Extract A and B logprobs
    logprob_A = logprobs_dict[" A"]
    logprob_B = logprobs_dict[" B"]
    
    # Convert to raw probabilities
    prob_A = math.exp(logprob_A)
    prob_B = math.exp(logprob_B)
    
    # Normalize to get distribution between just A and B
    total = prob_A + prob_B
    prob_A_normalized = prob_A / total
    prob_B_normalized = prob_B / total
    
    return {
        "A": {
            "logprob": logprob_A,
            "raw_prob": prob_A,
            "normalized_prob": prob_A_normalized
        },
        "B": {
            "logprob": logprob_B,
            "raw_prob": prob_B,
            "normalized_prob": prob_B_normalized
        }
    }

prompt = """SAMPLE A:
People like to draw boundaries. A boundary enables the description of elements that lie within and without the boundary. Drawing boundaries also has other effects:
- Boundaries can frame thoughts and hence lead to consideration of the inner or the outer without considering the whole. (are you with us or against us?)
- Boundaries allow you to characterise the nature of the boundary, i.e. permeable, semi-permeable, flexible, fixed, etc.
Brian Maricks recent exposition on testing metaphors
includes a boundary, that between the Agile team
and the rest of the world. He uses the following image
and says:They - programmers, testers, business experts - are in the business of protecting and nurturing the growing work until it's ready to face the world
I thinks the choice of boundary is interesting, Brian's choice is large, solid, horned defenders of the growing work
. This team looks independent, defends its young and I imagine it would survive outside of its current host organisation. That doesn't sound like the software projects I'm familiar with. In my experience (which includes software consultancy/service companies and niche product houses), projects tend to not have an existence outside their host organisation. Occasionally teams move to a new host, but it is uncommon. I don't think Brian intends for the team to be independant from its surroundings, but I can see how people may interpret it in this way. This is the problem with drawing boundaries and not discussing the nature of the boundary.
Maybe the testing metaphor (which is really a team metaphor) needs to acknowledge the interconnectedness of the team with the host organisation. I think the metaphor needs to conflict team identity interconnectness and symbiosis (though it has <REGIONAL_SECTION>been disputed, some projects are permanent parts of the host). Maybe viewing yourself as a metaphor, they have a strong centre, they are nourished with mine tea, logical and well, communication down to the detail stick the forehead communication paths. Remarkable to contemplate that the team are part of the whole. Still, it's an interesting idea. I like to explore the ideas of the independent and the interdependent, where the host organisation is also customer and where value can be created for both parties. In my experience, as an independent software developer, the client organisation is both of these things, they both create value for the software developer and the client, though I consider the client as the secondary part. This is why when I work with organisations to form strategic epics and tactics
(aka epics and strategies in the SmartBoardEPIC example
) is to create a context for the product, a context which is shared by the client and the product team, where both parties can work for mutual benefit.
That system metaphor is based on an assumption, that in a dependent and, to some extent, ordered software agile project, we have one software product to produce</REGIONAL_SECTION>. I think this assumption is incorrect. Team members can be part of a gelled team and yet have independance of thought and desire to find and fix defects. One of the developers I respect is extremely independant of thought, focuses in on defects with a laser accuracy, but can be seen pulling in the same direction as the other team members. If I had an agile team, there would be a place for this person, I don't want 'yes men', group polarisation and group think.
So, I'd like to see a better metaphor, but agree entirely with the article when it talks about testing and Is there an alternate metaphor that we can build upon? One that works with trust and close teamwork, rather than independently of them? Can we minimize the need for the independent and critical judge?
. I see trust, close teamwork and independance and critical thought working well in teams already, we don't need to abandon those things, just nurture them at the same time as we nurture the product.

SAMPLE B:
People like to draw boundaries. A boundary enables the description of elements that lie within and without the boundary. Drawing boundaries also has other effects:
- Boundaries can frame thoughts and hence lead to consideration of the inner or the outer without considering the whole. (are you with us or against us?)
- Boundaries allow you to characterise the nature of the boundary, i.e. permeable, semi-permeable, flexible, fixed, etc.
Brian Maricks recent exposition on testing metaphors
includes a boundary, that between the Agile team
and the rest of the world. He uses the following image
and says:They - programmers, testers, business experts - are in the business of protecting and nurturing the growing work until it's ready to face the world
I thinks the choice of boundary is interesting, Brian's choice is large, solid, horned defenders of the growing work
. This team looks independent, defends its young and I imagine it would survive outside of its current host organisation. That doesn't sound like the software projects I'm familiar with. In my experience (which includes software consultancy/service companies and niche product houses), projects tend to not have an existence outside their host organisation. Occasionally teams move to a new host, but it is uncommon. I don't think Brian intends for the team to be independant from its surroundings, but I can see how people may interpret it in this way. This is the problem with drawing boundaries and not discussing the nature of the boundary.
Maybe the testing metaphor (which is really a team metaphor) needs to acknowledge the interconnectedness of the team with the host organisation. I think the metaphor needs to conflate team identity, interconnectedness and symbiosis (though it has <REGIONAL_SECTION>to be admitted, some projects are parasites that kill the host). Maybe neurons work as a metaphor, they have a strong centre, they are connected via fine tendrils into the host, communications down particular links reinforces communication paths and they work to form a whole that is greater than the sum of the parts. Initially attracted to that idea, I don't see it as very successful. Brian's desire seems to be to bring independance, critical thought and error seeking behavior inside rather than outside the team is laudable, and isn't addressed. However, the dinosaur ring metaphor chosen is akin to protecting the child against all comers, seeking to nurture it but not allowing it outside until it is grown. Is this really the right way? Shouldn't the child be encouraged into the open early, supported but not closeted, exposed to new people and ideas and encouraged to explore? Social networks are a better metaphor for software teams. There can be a place for independence, critical thought, error seeking and cohesiveness within social networks. Social networks also capture the lumpy nature of teams and cohesion, within large teams there are clumps of high cohesion and within those clumps there are other clumps, it's fractal in nature (to some extent anyway).
Maybe the dinosaur ring is born from the statement that independance and - to some extent - error-seeking are not a natural fit for Agile projects, which thrive on close teamwork and trust
. This statement is based on an assumption, that independance and, to some extent, error seeking, restrict close teamwork and trust</REGIONAL_SECTION>. I think this assumption is incorrect. Team members can be part of a gelled team and yet have independance of thought and desire to find and fix defects. One of the developers I respect is extremely independant of thought, focuses in on defects with a laser accuracy, but can be seen pulling in the same direction as the other team members. If I had an agile team, there would be a place for this person, I don't want 'yes men', group polarisation and group think. Despite the pro group-think article
, most teams that attempt it will fail. The effect on both the teams judgement and the host will be too detrimental.
So, I'd like to see a better metaphor, but agree entirely with the article when it talks about testing and Is there an alternate metaphor that we can build upon? One that works with trust and close teamwork, rather than independently of them? Can we minimize the need for the independent and critical judge?
. I see trust, close teamwork and independance and critical thought working well in teams already, we don't need to abandon those things, just nurture them at the same time as we nurture the product.

ANSWER:"""

response = query_vllm(prompt)
top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
distribution = logprobs_to_probs(top_logprobs)

print("\nProbability Distribution between A and B:")
print(f"A: {distribution['A']['normalized_prob']:.2%}")
print(f"B: {distribution['B']['normalized_prob']:.2%}")
print("\nFull details:")
print(json.dumps(distribution, indent=2))