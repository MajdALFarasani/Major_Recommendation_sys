CATEGORY_MAP = {
        'relaxation_wellness':    {'meditation', 'yoga', 'birdwatching', 'tarot reading'},
        
        'music_performance':      {'singing','playing musical instruments'},
        
        'stage_performance':      {'stand-up comedy','magic'},
        
        'culinary_tasting':       {'cooking', 'baking', 'homebrewing', 'wine tasting'},
        
        'intellectual_research':  {'reading', 'writing', 'programming',
                                   'numismatics', 'research'},
        
        'community_service':      {'volunteering','philanthropy'},
        
        'social_games':           {'board games','card games','puzzle solving','escape rooms'},
        

        # Trait-based sports categories
        'trait_teamwork_leadership': {
            'soccer', 'basketball', 'water polo', 'handball', 'disc golf', 'dodgeball','sports'
        },
        'trait_endurance_grit': {
            'running', 'jogging', 'cycling', 'swimming', 'skiing'
        },
        'trait_strength_resilience': {
            'weightlifting', 'rock climbing', 'bouldering'
        },
        'trait_strategic_thinking': {
            'disc golf', 'dodgeball', 'parkour', 'fencing'
        },
        'trait_precision_coordination': {
            'archery', 'table tennis', 'badminton'
        },
        'trait_risk_adaptability': {
            'skydiving', 'rafting', 'snowboarding',
            'scuba diving', 'snorkeling'
        },
        # Visual & tactile arts
        'visual_arts': {
        'painting', 'drawing', 'graphic design', 'fashion design', 'cosplay'
        },
        'literary_arts': {
            'poetry', 'calligraphy', 'writing'
        },
        'functional_crafts': {
            'pottery', 'quilting', 'knitting', 'woodworking',
            'candle making', 'soap making', 'glassblowing', 'leatherworking', 'scrapbooking'
        },

        # Media creation vs. consumption
        'media_creation': {
            'video editing', 'vlogging', 'podcasting', 'photo editing'
        },
        'gaming_entertainment': {
            'gaming'
        },

        'collecting_hobbies':{
            'collecting (e.g., stamps, coins, comic books)'
        },

        # Tech & maker
        'tech_building': {
            'robotics', 'programming', 'web development', 'drone piloting'
        }
    }

updates = {
    'relaxation_wellness': {
        'bird watching', 'bird training', 'foraging',
        'gardening', 'stargazing', 'tai chi', 'traveling'
    },
    'stage_performance': {
        'acting', 'dancing', 'larping', 'puppetry'
    },
    'music_performance': {'music'},
    'visual_arts': {'animation', 'digital art', 'makeup artistry'},
    'functional_crafts': {
        'beekeeping', 'diy projects', 'jewelry making', 'model building',
        'origami', 'stone carving', 'upcycling', 'whittling'
    },
    'media_creation': {'photography'},
    'intellectual_research': {
        'astrology', 'astronomy', 'collecting', 'philately', 'metal detecting'
    },
    'community_service': {'animal rescue'},
    'trait_endurance_grit': {
        'biking', 'camping', 'fishing', 'hiking', 'mountain biking', 'zumba'
    },
    'trait_strength_resilience': {
        'callisthenics', 'fitness training', 'kickboxing', 'martial arts'
    },
    'trait_precision_coordination': {
        'horseback riding', 'ice skating', 'roller skating'
    },
    'trait_risk_adaptability': {
        'flying planes', 'freerunning', 'kayaking', 'sailing', 'skateboarding',
        'spelunking', 'surfing', 'urban exploration', 'windsurfing'
    },
    'trait_strategic_thinking': {'geocaching'},
    }
    
for cat, hobbies in updates.items():
    CATEGORY_MAP.setdefault(cat, set()).update(hobbies)
